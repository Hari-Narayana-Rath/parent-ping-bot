from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import extra_streamlit_components as stx
import streamlit as st


CAMERA_COOKIE_KEY = "pp_camera_settings_v1"
COOKIE_DAYS = 30


def _kill_process(pid: int) -> None:
    if pid <= 0:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        pass


def _load_settings(cookie_manager: stx.CookieManager, defaults: dict[str, str]) -> dict[str, str]:
    raw = cookie_manager.get(CAMERA_COOKIE_KEY)
    if not raw:
        return dict(defaults)
    try:
        data = json.loads(raw)
        merged = dict(defaults)
        for key in defaults:
            if key in data and isinstance(data[key], str) and data[key].strip():
                merged[key] = data[key].strip()
        return merged
    except Exception:
        return dict(defaults)


def _save_settings(cookie_manager: stx.CookieManager, settings: dict[str, str]) -> None:
    try:
        cookie_manager.set(
            CAMERA_COOKIE_KEY,
            json.dumps(settings),
            expires_at=datetime.now(timezone.utc) + timedelta(days=COOKIE_DAYS),
        )
    except Exception:
        pass


def run_camera_portal() -> None:
    st.set_page_config(page_title="ParentPing Camera App", layout="centered")
    st.title("ParentPing Camera App")
    st.caption("Run this only on the classroom PC with a webcam. Parents use the Parent Portal on another device.")

    st.warning(
        "Hosted Streamlit cannot access your classroom webcam. "
        "Run this page locally (`streamlit run camera_portal.py`) on the machine where the camera is plugged in."
    )

    cookie_manager = stx.CookieManager(key="parentping_camera_cookie_v1")

    default_api = os.getenv("PARENTPING_API_BASE_URL", "http://127.0.0.1:8000")
    defaults = {
        "api_url": default_api,
        "weights_path": "best_resnet18_arcface_parentping.pth",
        "db_path": "parentping.db",
        "camera_secret": os.getenv("PARENTPING_CAMERA_SECRET", ""),
    }
    settings = _load_settings(cookie_manager, defaults)

    if "camera_pid" not in st.session_state:
        st.session_state.camera_pid = None

    api_url = st.text_input("Backend API URL (Render or local)", value=settings["api_url"])
    weights_path = st.text_input("Model Weights Path", value=settings["weights_path"])
    db_path = st.text_input("Local Database Path", value=settings["db_path"])
    camera_secret = st.text_input(
        "Camera secret (must match PARENTPING_CAMERA_SECRET on Render)",
        value=settings.get("camera_secret", ""),
        type="password",
        help="Set the same random string in your API environment on Render. Required for live “Student in classroom” from the webcam.",
    )
    max_faces = st.slider("Max faces to track per frame", min_value=1, max_value=6, value=3)

    col_start, col_stop = st.columns(2)
    with col_start:
        start = st.button("Start camera service", type="primary")
    with col_stop:
        stop = st.button("Stop camera service")

    if start:
        _save_settings(
            cookie_manager,
            {
                "api_url": api_url.strip(),
                "weights_path": weights_path.strip(),
                "db_path": db_path.strip(),
                "camera_secret": camera_secret.strip(),
            },
        )
        if st.session_state.camera_pid:
            _kill_process(st.session_state.camera_pid)
            st.session_state.camera_pid = None
        try:
            env = os.environ.copy()
            if camera_secret.strip():
                env["PARENTPING_CAMERA_SECRET"] = camera_secret.strip()
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "parentping.camera.realtime_camera",
                    "--weights",
                    weights_path.strip(),
                    "--db",
                    db_path.strip(),
                    "--api",
                    api_url.strip().rstrip("/"),
                    "--max-faces",
                    str(int(max_faces)),
                    "--camera-secret",
                    camera_secret.strip(),
                ],
                env=env,
            )
            st.session_state.camera_pid = proc.pid
            st.success(f"Camera service started (PID {proc.pid}). You can refresh this page without losing these settings.")
        except Exception as exc:
            st.error(f"Could not start camera: {exc}")

    if stop:
        if st.session_state.camera_pid:
            _kill_process(st.session_state.camera_pid)
            st.session_state.camera_pid = None
            st.success("Stop signal sent to the camera process.")
        else:
            st.info("No camera PID tracked in this session. If it is still running, close the OpenCV window or use Task Manager.")

    pid_note = st.session_state.camera_pid
    if pid_note:
        st.caption(f"Active camera process PID: **{pid_note}** — use **Stop** before starting again.")

    st.code(
        f'{sys.executable} -m parentping.camera.realtime_camera --weights "{weights_path}" --db "{db_path}" '
        f'--api "{api_url}" --max-faces {int(max_faces)} --camera-secret "<your-secret>"',
        language="bash",
    )


if __name__ == "__main__":
    run_camera_portal()
