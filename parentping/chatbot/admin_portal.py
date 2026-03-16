from __future__ import annotations

import requests
import streamlit as st


st.set_page_config(page_title="ParentPing Admin", layout="centered")
st.title("ParentPing Admin Registration")
st.caption("Register a new student using a multi-angle face video")

api_base = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000").rstrip("/")

with st.form("register_video_form"):
    name = st.text_input("Student Name")
    roll_number = st.text_input("Roll Number")
    parent_email = st.text_input("Parent Email")
    parent_password = st.text_input("Parent Password", type="password")
    video_file = st.file_uploader(
        "Student Face Video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Upload a clear 10-30 second video covering multiple face angles.",
    )
    submitted = st.form_submit_button("Register Student")

if submitted:
    if not all([name.strip(), roll_number.strip(), parent_email.strip(), parent_password.strip(), video_file]):
        st.error("All fields and a video file are required.")
    else:
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
                f"{api_base}/register_student_from_video",
                data=data,
                files=files,
                timeout=120,
            )
            if response.ok:
                payload = response.json()
                st.success(
                    f"Registered successfully. Student ID: {payload.get('student_id')} | "
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

