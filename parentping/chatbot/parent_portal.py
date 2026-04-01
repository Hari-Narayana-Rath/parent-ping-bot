from __future__ import annotations

import datetime as dt
import json
import os
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import extra_streamlit_components as stx
import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("PARENTPING_API_BASE_URL", "https://parentping-api.onrender.com")
REQUEST_TIMEOUT_SECONDS = 75
SESSION_COOKIE_KEY = "pp_parent_session_v1"
POLL_INTERVAL = dt.timedelta(seconds=10)
COOKIE_DAYS = 14


def _clean_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _is_valid_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _request_json(
    method: str,
    path: str,
    token: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    resolved_base_url = _clean_base_url(base_url or st.session_state.api_base_url)
    if not resolved_base_url:
        raise RuntimeError("API Base URL is not configured.")
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{resolved_base_url}{path}"
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
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = _clean_base_url(DEFAULT_API_BASE_URL)
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
    if "_session_restored" not in st.session_state:
        st.session_state._session_restored = False


def _cookie_expires() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=COOKIE_DAYS)


def _persist_session_cookie(cookie_manager: stx.CookieManager) -> None:
    payload = {
        "token": st.session_state.parent_token,
        "student_id": st.session_state.student_id,
        "student_name": st.session_state.student_name,
        "roll_number": st.session_state.roll_number,
        "api_base_url": st.session_state.api_base_url,
    }
    try:
        cookie_manager.set(
            SESSION_COOKIE_KEY,
            json.dumps(payload),
            expires_at=_cookie_expires(),
        )
    except Exception:
        pass


def _clear_session_cookie(cookie_manager: stx.CookieManager) -> None:
    try:
        cookie_manager.delete(SESSION_COOKIE_KEY)
    except Exception:
        pass


def _apply_session_payload(data: Dict[str, Any]) -> None:
    st.session_state.parent_token = str(data.get("token") or "")
    sid = data.get("student_id")
    st.session_state.student_id = int(sid) if sid is not None else None
    st.session_state.student_name = str(data.get("student_name") or "")
    st.session_state.roll_number = str(data.get("roll_number") or "")
    api = data.get("api_base_url")
    if api and _is_valid_http_url(_clean_base_url(str(api))):
        st.session_state.api_base_url = _clean_base_url(str(api))


def _try_restore_session_from_cookie(cookie_manager: stx.CookieManager) -> None:
    if st.session_state.parent_token or st.session_state._session_restored:
        return
    raw = cookie_manager.get(SESSION_COOKIE_KEY)
    if not raw:
        st.session_state._session_restored = True
        return
    try:
        data = json.loads(raw)
    except Exception:
        st.session_state._session_restored = True
        return
    if not data.get("token") or data.get("student_id") is None:
        st.session_state._session_restored = True
        return
    _apply_session_payload(data)
    try:
        sid = int(st.session_state.student_id)
        _request_json("GET", f"/attendance/{sid}/today", token=st.session_state.parent_token)
    except Exception:
        st.session_state.parent_token = ""
        st.session_state.student_id = None
        st.session_state.student_name = ""
        st.session_state.roll_number = ""
        _clear_session_cookie(cookie_manager)
    st.session_state._session_restored = True


def _send_query(query: str) -> None:
    st.session_state.chat_messages.append(("parent", query))
    try:
        response = _post_json("/chatbot_query", {"query": query}, token=st.session_state.parent_token)
        st.session_state.chat_messages.append(("assistant", response.get("response", "")))
    except Exception as exc:
        st.session_state.chat_messages.append(("assistant", f"Error: {exc}"))


def _fetch_today_snapshot() -> Optional[Dict[str, Any]]:
    if not st.session_state.student_id or not st.session_state.parent_token:
        return None
    try:
        return _get_json(
            f"/attendance/{st.session_state.student_id}/today",
            token=st.session_state.parent_token,
        )
    except Exception:
        return None


def _snapshot_to_classroom_label(snap: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """in_class is True only when checked in today and not yet checked out (API record)."""
    if not snap:
        return False, "Unknown"
    if not snap.get("has_record"):
        return False, "Not checked in yet"
    if snap.get("in_class"):
        return True, "Yes — checked in"
    return False, "Checked out"


def _fragment_supported() -> bool:
    return hasattr(st, "fragment")


def run_app() -> None:
    _init_state()

    st.set_page_config(page_title="ParentPing Chat Bot", layout="wide")
    st.title("ParentPing Chat Bot")
    st.caption(
        "Parent-only attendance assistant — session persists across reloads; status updates automatically. "
        "**In classroom** uses the attendance record from the camera (check-in / check-out), not live video."
    )

    st.markdown(
        """
        <style>
        .status-card {
          border: 1px solid #1e293b;
          border-radius: 12px;
          padding: 12px 14px;
          background: linear-gradient(180deg, #0f172a, #111827);
          color: #f8fafc;
        }
        .status-card strong { color: #f8fafc; }
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

    cookie_manager = stx.CookieManager(key="parentping_parent_cookie_v1")
    _try_restore_session_from_cookie(cookie_manager)

    with st.sidebar:
        st.markdown("### Connection")
        api_input = st.text_input("Backend API URL", value=st.session_state.api_base_url)
        if st.button("Update API URL"):
            cleaned_url = _clean_base_url(api_input)
            if not _is_valid_http_url(cleaned_url):
                st.error("Enter a valid URL (http/https).")
            else:
                st.session_state.api_base_url = cleaned_url
                if st.session_state.parent_token:
                    _persist_session_cookie(cookie_manager)
                st.success("Backend URL updated.")
                st.rerun()
        st.caption(f"Current: `{st.session_state.api_base_url or 'Not set'}`")
        st.caption("Tip: Open this app on your phone; run the camera app only on the classroom PC.")

    if not st.session_state.api_base_url:
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
                    _persist_session_cookie(cookie_manager)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.info("Use the student roll number and the password given by the admin. Your session is saved in this browser.")
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
            _clear_session_cookie(cookie_manager)
            st.rerun()

    def _render_status_card(snap: Optional[Dict[str, Any]]) -> None:
        in_class, label = _snapshot_to_classroom_label(snap)
        color = "#1f9d55" if in_class else ("#64748b" if "Not checked" in label else "#e03131")
        extra = ""
        if snap and snap.get("has_record") and snap.get("time_in"):
            tin = snap.get("time_in", "")
            tout = snap.get("time_out") or "—"
            extra = f"<br/><small>In: {tin} · Out: {tout}</small>"
        st.markdown(
            f"""
            <div class="status-card">
              <span class="status-dot" style="background:{color};"></span>
              <strong>Student In Classroom:</strong> {label}{extra}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if _fragment_supported():
        @st.fragment(run_every=POLL_INTERVAL)
        def _live_status() -> None:
            snap = _fetch_today_snapshot()
            _render_status_card(snap)

        _live_status()
        st.caption(f"Status refreshes about every {int(POLL_INTERVAL.total_seconds())} seconds from the API.")
    else:
        snap = _fetch_today_snapshot()
        _render_status_card(snap)
        if st.button("Refresh classroom status"):
            st.rerun()

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
