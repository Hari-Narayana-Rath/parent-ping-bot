from __future__ import annotations

import datetime as dt
import time
from typing import Any, Dict

import requests
import streamlit as st


API_BASE_URL = "https://parentping-api.onrender.com"
REQUEST_TIMEOUT_SECONDS = 75


def _request_json(method: str, path: str, token: str | None = None, **kwargs: Any) -> Any:
    if not API_BASE_URL:
        raise RuntimeError("API Base URL is not configured.")
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{API_BASE_URL}{path}"
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
            if not response.ok:
                detail = response.text
                try:
                    detail = response.json().get("detail", detail)
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {response.status_code}: {detail}")
            return response.json()
        except requests.exceptions.Timeout as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(3)
                continue
        except Exception as exc:
            last_error = exc
            break

    if isinstance(last_error, requests.exceptions.Timeout):
        raise RuntimeError(
            "The backend is taking too long to respond. "
            "If Render is on the free tier, wait about a minute and try again."
        )
    if last_error:
        raise RuntimeError(str(last_error))
    raise RuntimeError("Unknown request failure.")


def _post_json(path: str, payload: Dict[str, Any], token: str | None = None) -> Dict[str, Any]:
    return _request_json("POST", path, token=token, json=payload)


def _get_json(path: str, token: str | None = None) -> Any:
    return _request_json("GET", path, token=token)


def _init_state() -> None:
    if "parent_token" not in st.session_state:
        st.session_state.parent_token = ""
    if "student_id" not in st.session_state:
        st.session_state.student_id = None
    if "student_name" not in st.session_state:
        st.session_state.student_name = ""
    if "roll_number" not in st.session_state:
        st.session_state.roll_number = ""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def _send_query(query: str) -> None:
    st.session_state.chat_messages.append(("parent", query))
    try:
        response = _post_json("/chatbot_query", {"query": query}, token=st.session_state.parent_token)
        st.session_state.chat_messages.append(("assistant", response.get("response", "")))
    except Exception as exc:
        st.session_state.chat_messages.append(("assistant", f"Error: {exc}"))


def _get_classroom_status() -> tuple[bool, str]:
    if not st.session_state.student_id:
        return False, "No"
    try:
        records = _get_json(
            f"/attendance/{st.session_state.student_id}",
            token=st.session_state.parent_token,
        )
    except Exception:
        return False, "No"

    today = dt.date.today().isoformat()
    for record in records:
        if record.get("date") == today:
            in_class = record.get("time_out") in (None, "", "null")
            return in_class, "Yes" if in_class else "No"
    return False, "No"


def run_app() -> None:
    _init_state()

    st.set_page_config(page_title="ParentPing Chat Bot", layout="wide")
    st.title("ParentPing Chat Bot")
    st.caption("Parent-only attendance assistant")

    st.markdown(
        """
        <style>
        .status-card {
          border: 1px solid #dfe5ec;
          border-radius: 12px;
          padding: 12px 14px;
          background: linear-gradient(180deg, #ffffff, #f8fafc);
        }
        .status-dot {
          display: inline-block;
          width: 10px;
          height: 10px;
          border-radius: 50%;
          margin-right: 8px;
          animation: pulse 1.6s infinite;
        }
        @keyframes pulse {
          0% { transform: scale(0.9); opacity: 0.9; }
          50% { transform: scale(1.1); opacity: 0.65; }
          100% { transform: scale(0.9); opacity: 0.9; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not API_BASE_URL:
        st.error("Application backend is not configured. Contact the administrator.")
        return

    if not st.session_state.parent_token:
        with st.form("parent_login_form", clear_on_submit=False):
            roll_number = st.text_input("Student Roll Number")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                try:
                    result = _post_json(
                        "/login_parent",
                        {"roll_number": roll_number, "password": password},
                    )
                    st.session_state.parent_token = result["access_token"]
                    st.session_state.student_id = result.get("student_id")
                    st.session_state.student_name = result.get("student_name") or ""
                    st.session_state.roll_number = result.get("roll_number") or ""
                    st.session_state.chat_messages = [
                        ("assistant", "Login successful. You can now ask about your ward's attendance.")
                    ]
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.info("Use the student roll number and the password given by the admin.")
        return

    head_left, head_right = st.columns([4, 1])
    with head_left:
        st.markdown(
            f"**Ward:** {st.session_state.student_name or 'N/A'}  \n"
            f"**Roll Number:** {st.session_state.roll_number or 'N/A'}"
        )
    with head_right:
        if st.button("Logout"):
            st.session_state.parent_token = ""
            st.session_state.student_id = None
            st.session_state.student_name = ""
            st.session_state.roll_number = ""
            st.session_state.chat_messages = []
            st.rerun()

    in_class, label = _get_classroom_status()
    color = "#1f9d55" if in_class else "#e03131"
    st.markdown(
        f"""
        <div class="status-card">
          <span class="status-dot" style="background:{color};"></span>
          <strong>Student In Classroom:</strong> {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    prompt_cols = st.columns(4)
    prompts = [
        "Did my child attend class today?",
        "What time did my child enter the classroom today?",
        "What time did my child leave the classroom today?",
        "Show this week's attendance.",
    ]
    for index, prompt in enumerate(prompts):
        with prompt_cols[index]:
            if st.button(prompt, key=f"parent_prompt_{index}"):
                _send_query(prompt)
                st.rerun()

    for role, message in st.session_state.chat_messages:
        with st.chat_message("user" if role == "parent" else "assistant"):
            st.markdown(message)

    user_query = st.chat_input("Ask about your child's attendance")
    if user_query and user_query.strip():
        _send_query(user_query.strip())
        st.rerun()


if __name__ == "__main__":
    run_app()
