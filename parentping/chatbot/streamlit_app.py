from __future__ import annotations

import datetime as dt
from typing import Any, Dict

import requests
import streamlit as st


st.set_page_config(page_title="ParentPing Portal", layout="wide")
st.title("ParentPing Portal")
st.caption("Parent chatbot and student registration")

API_BASE_URL = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000").rstrip("/")
mode = st.sidebar.radio("Mode", ["Parent Chatbot", "Admin Registration"])
st.markdown(
    """
<style>
.status-card {
  border: 1px solid #dfe5ec;
  border-radius: 12px;
  padding: 12px 14px;
  background: linear-gradient(180deg, #ffffff, #f8fafc);
  animation: fadeIn 0.5s ease-out;
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
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
    """,
    unsafe_allow_html=True,
)


def _post_json(path: str, payload: Dict[str, Any], token: str | None = None) -> Dict[str, Any]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        headers=headers,
        timeout=20,
    )
    if not response.ok:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    return response.json()


def _get_json(path: str, token: str | None = None) -> Any:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(
        f"{API_BASE_URL}{path}",
        headers=headers,
        timeout=20,
    )
    if not response.ok:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    return response.json()


if "token" not in st.session_state:
    st.session_state.token = ""
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
        response = _post_json(
            "/chatbot_query",
            {"query": query},
            token=st.session_state.token,
        )
        st.session_state.chat_messages.append(("assistant", response.get("response", "")))
    except Exception as exc:
        st.session_state.chat_messages.append(("assistant", f"Error: {exc}"))


def _get_classroom_status() -> tuple[bool, str]:
    student_id = st.session_state.student_id
    if not student_id:
        return False, "No"
    try:
        records = _get_json(f"/attendance/{student_id}", token=st.session_state.token)
    except Exception:
        return False, "No"
    if not records:
        return False, "No"

    today = dt.date.today().isoformat()
    latest_today = None
    for rec in records:
        if rec.get("date") == today:
            latest_today = rec
            break
    if not latest_today:
        return False, "No"

    in_class = latest_today.get("time_out") in (None, "", "null")
    return in_class, "Yes" if in_class else "No"


def _render_parent_chatbot() -> None:
    st.subheader("Parent Chatbot")
    if not st.session_state.token:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Parent Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                try:
                    result = _post_json("/login_parent", {"email": email, "password": password})
                    st.session_state.token = result["access_token"]
                    st.session_state.student_id = result.get("student_id")
                    st.session_state.student_name = result.get("student_name") or ""
                    st.session_state.roll_number = result.get("roll_number") or ""
                    st.session_state.chat_messages = [
                        ("assistant", "You are now logged in. Ask attendance questions about your child.")
                    ]
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.info("Login is required to access chat.")
        return

    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            f"**Student:** {st.session_state.student_name or 'N/A'}  \n"
            f"**Roll Number:** {st.session_state.roll_number or 'N/A'}"
        )
    with right:
        if st.button("Logout"):
            st.session_state.token = ""
            st.session_state.student_id = None
            st.session_state.student_name = ""
            st.session_state.roll_number = ""
            st.session_state.chat_messages = []
            st.rerun()

    in_class, status_text = _get_classroom_status()
    status_color = "#1f9d55" if in_class else "#e03131"
    st.markdown(
        f"""
        <div class="status-card">
          <span class="status-dot" style="background:{status_color};"></span>
          <strong>Student In Classroom:</strong> {status_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Quick Prompts**")
    prompts = [
        "Did my child attend class today?",
        "What time did my child enter the classroom today?",
        "What time did my child leave the classroom today?",
        "Show this week's attendance.",
    ]
    cols = st.columns(4)
    for i, prompt in enumerate(prompts):
        with cols[i % 4]:
            if st.button(prompt, key=f"quick_prompt_{i}"):
                _send_query(prompt)
                st.rerun()

    st.markdown("**Chat**")
    for role, message in st.session_state.chat_messages:
        with st.chat_message("user" if role == "parent" else "assistant"):
            st.markdown(message)

    user_query = st.chat_input("Ask about your child's attendance")
    if user_query and user_query.strip():
        _send_query(user_query.strip())
        st.rerun()


def _render_admin_registration() -> None:
    st.subheader("Admin Registration")
    st.caption("Register a student using a multi-angle face video")
    with st.form("register_video_form"):
        name = st.text_input("Student Name")
        roll_number = st.text_input("Roll Number")
        parent_email = st.text_input("Parent Email")
        parent_password = st.text_input("Parent Password", type="password")
        video_file = st.file_uploader(
            "Student Face Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload a clear multi-angle 10-30 second video.",
        )
        submitted = st.form_submit_button("Register Student")

    if submitted:
        if not all([name.strip(), roll_number.strip(), parent_email.strip(), parent_password.strip(), video_file]):
            st.error("All fields and a video file are required.")
            return

        files = {
            "video": (video_file.name, video_file.getvalue(), video_file.type or "video/mp4"),
        }
        data = {
            "name": name.strip(),
            "roll_number": roll_number.strip(),
            "parent_email": parent_email.strip(),
            "parent_password": parent_password,
        }
        try:
            response = requests.post(
                f"{API_BASE_URL}/register_student_from_video",
                data=data,
                files=files,
                timeout=180,
            )
            if response.ok:
                payload = response.json()
                st.success(
                    f"Registered successfully | Student ID: {payload.get('student_id')} | "
                    f"Parent ID: {payload.get('parent_id')}"
                )
            else:
                detail = response.text
                try:
                    detail = response.json().get("detail", detail)
                except Exception:
                    pass
                st.error(f"Registration failed: {detail}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")


if mode == "Parent Chatbot":
    _render_parent_chatbot()
else:
    _render_admin_registration()
