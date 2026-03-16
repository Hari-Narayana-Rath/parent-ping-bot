from __future__ import annotations

import os
import subprocess
import sys

import streamlit as st


st.set_page_config(page_title="ParentPing Camera App", layout="centered")
st.title("ParentPing Camera App")
st.caption("Dedicated camera-side attendance launcher")

st.warning(
    "This app is intended for the classroom machine. "
    "Hosted Streamlit environments cannot use the classroom webcam for your OpenCV attendance loop."
)

default_api_url = os.getenv("PARENTPING_API_BASE_URL", "http://127.0.0.1:8000")
api_url = st.text_input("Backend API URL", value=default_api_url)
weights_path = st.text_input("Model Weights Path", value="best_resnet18_arcface_parentping.pth")
db_path = st.text_input("Local Database Path", value="parentping.db")

if st.button("Start Camera Service"):
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "parentping.camera.realtime_camera",
            "--weights",
            weights_path,
            "--db",
            db_path,
            "--api",
            api_url,
        ]
    )
    st.success("Camera service launched in a separate process on this machine.")

st.code(
    f'{sys.executable} -m parentping.camera.realtime_camera --weights "{weights_path}" --db "{db_path}" --api "{api_url}"',
    language="bash",
)
