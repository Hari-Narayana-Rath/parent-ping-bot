from __future__ import annotations

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


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("detail", response.text))
    except Exception:
        pass
    return response.text.strip() or f"HTTP {response.status_code}"


def _post_json(path: str, payload: Dict[str, Any], token: str | None = None) -> Dict[str, Any]:
    return _request_json("POST", path, token=token, json=payload)


def _get_json(path: str, token: str | None = None) -> Any:
    return _request_json("GET", path, token=token)


def _delete_student(student_id: int, token: str) -> str:
    response = _request_json("DELETE", f"/admin/student/{student_id}", token=token)
    return response.get("message", "Student removed successfully.")


def _init_state() -> None:
    if "admin_token" not in st.session_state:
        st.session_state.admin_token = ""


def run_app() -> None:
    _init_state()

    st.set_page_config(page_title="ParentPing Admin", layout="wide")
    st.title("ParentPing Admin")
    st.caption("Admin-only student management portal")

    if not API_BASE_URL:
        st.error("Application backend is not configured.")
        return

    if not st.session_state.admin_token:
        with st.form("admin_login_form", clear_on_submit=False):
            email = st.text_input("Admin Email")
            password = st.text_input("Admin Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                try:
                    result = _post_json("/login_admin", {"email": email, "password": password})
                    st.session_state.admin_token = result["access_token"]
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        return

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown("**Admin session active**")
    with top_right:
        if st.button("Logout"):
            st.session_state.admin_token = ""
            st.rerun()

    with st.expander("Upload Model File", expanded=False):
        model_file = st.file_uploader("ArcFace Model (.pth)", type=["pth"], key="admin_model_file")
        if st.button("Upload Model", key="upload_model_button"):
            if not model_file:
                st.error("Select a model file.")
            else:
                response = requests.post(
                    f"{API_BASE_URL}/admin/upload_model",
                    headers={"Authorization": f"Bearer {st.session_state.admin_token}"},
                    files={"model_file": (model_file.name, model_file.getvalue(), "application/octet-stream")},
                    timeout=300,
                )
                if response.ok:
                    st.success(response.json().get("message", "Model uploaded successfully."))
                else:
                    st.error(_response_detail(response))

    with st.expander("Import Private Data", expanded=False):
        data_file = st.file_uploader("Private Data Export (.json)", type=["json"], key="admin_import_file")
        replace_existing = st.checkbox("Replace existing hosted data", key="replace_existing_data")
        if st.button("Import Data", key="import_data_button"):
            if not data_file:
                st.error("Select a JSON export file.")
            else:
                response = requests.post(
                    f"{API_BASE_URL}/admin/import_data",
                    headers={"Authorization": f"Bearer {st.session_state.admin_token}"},
                    files={"data_file": (data_file.name, data_file.getvalue(), "application/json")},
                    data={"replace_existing": str(replace_existing).lower()},
                    timeout=300,
                )
                if response.ok:
                    st.success(response.json().get("message", "Data imported successfully."))
                else:
                    st.error(_response_detail(response))

    st.markdown("**Register Student**")
    with st.form("register_student_form"):
        name = st.text_input("Student Name")
        roll_number = st.text_input("Roll Number")
        parent_email = st.text_input("Parent Email")
        parent_password = st.text_input("Parent Portal Password", type="password")
        video_file = st.file_uploader(
            "Student Face Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="admin_register_video",
        )
        submitted = st.form_submit_button("Register Student")
        if submitted:
            if not all([name.strip(), roll_number.strip(), parent_email.strip(), parent_password.strip(), video_file]):
                st.error("All fields and a student video are required.")
            else:
                response = requests.post(
                    f"{API_BASE_URL}/register_student_from_video",
                    headers={"Authorization": f"Bearer {st.session_state.admin_token}"},
                    data={
                        "name": name.strip(),
                        "roll_number": roll_number.strip(),
                        "parent_email": parent_email.strip(),
                        "parent_password": parent_password,
                    },
                    files={"video": (video_file.name, video_file.getvalue(), video_file.type or "video/mp4")},
                    timeout=300,
                )
                if response.ok:
                    st.success(response.json().get("message", "Student registered successfully."))
                else:
                    st.error(_response_detail(response))

    st.markdown("**Manage Students**")
    try:
        students = _get_json("/admin/students", token=st.session_state.admin_token)
    except Exception as exc:
        st.error(str(exc))
        return

    if not students:
        st.info("No students found.")
        return

    for student in students:
        student_id = int(student["id"])
        left, mid, right = st.columns([4, 2, 2])
        with left:
            st.markdown(
                f"**{student['name']}**  \n"
                f"Roll Number: `{student['roll_number']}`  \n"
                f"Parent Email: `{student['parent_email']}`"
            )
        with mid:
            new_password = st.text_input(
                f"New password for {student['roll_number']}",
                type="password",
                key=f"reset_pw_{student_id}",
            )
            if st.button("Change Password", key=f"reset_btn_{student_id}"):
                try:
                    result = _post_json(
                        "/admin/reset_parent_password",
                        {"student_id": student_id, "new_password": new_password},
                        token=st.session_state.admin_token,
                    )
                    st.success(result["message"])
                except Exception as exc:
                    st.error(str(exc))
        with right:
            if st.button("Remove Student", key=f"delete_student_{student_id}"):
                try:
                    message = _delete_student(student_id, st.session_state.admin_token)
                    st.success(message)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.divider()


if __name__ == "__main__":
    run_app()
