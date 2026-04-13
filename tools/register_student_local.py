from __future__ import annotations

import argparse
import getpass
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import requests

from parentping.models.embedding_model import load_embedding_model
from parentping.recognition.embedding_extractor import EmbeddingExtractor
from parentping.recognition.face_detector import FaceDetector


def _request_json(method: str, url: str, **kwargs: Any) -> Any:
    response = requests.request(method, url, timeout=90, **kwargs)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    return response.json()


def _embedding_from_video(
    *,
    video_path: Path,
    weights_path: Path,
    min_faces: int = 5,
    max_embeddings: int = 20,
) -> np.ndarray:
    model, device = load_embedding_model(weights_path)
    detector = FaceDetector(use_retinaface=False)
    extractor = EmbeddingExtractor(model=model, device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(total_frames // 80, 1)
    frame_index = 0
    embeddings: list[np.ndarray] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            detections = detector.detect_faces(frame)
            face = detector.crop_largest_face(frame, detections)
            if face is not None and face.size > 0:
                embeddings.append(extractor.extract(face))
                print(f"Collected face embedding {len(embeddings)}/{max_embeddings}", flush=True)
                if len(embeddings) >= max_embeddings:
                    break
            frame_index += 1
    finally:
        cap.release()

    if len(embeddings) < min_faces:
        raise RuntimeError(
            f"Only {len(embeddings)} clear face frames were found. "
            "Use a brighter, closer, multi-angle face video."
        )

    merged = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(merged))
    if norm == 0.0:
        raise RuntimeError("Generated embedding has zero norm.")
    return merged / norm


def register_student(args: argparse.Namespace) -> None:
    api_base_url = args.api.rstrip("/")
    admin_email = args.admin_email or input("Admin email: ").strip()
    admin_password = args.admin_password or getpass.getpass("Admin password: ")

    print("Logging in to admin API...", flush=True)
    login = _request_json(
        "POST",
        f"{api_base_url}/login_admin",
        json={"email": admin_email, "password": admin_password},
    )
    token = login["access_token"]

    print("Processing video locally. Render will not run PyTorch for this step.", flush=True)
    embedding = _embedding_from_video(
        video_path=Path(args.video).expanduser().resolve(),
        weights_path=Path(args.weights).expanduser().resolve(),
        min_faces=args.min_faces,
        max_embeddings=args.max_embeddings,
    )

    payload = {
        "name": args.name.strip(),
        "roll_number": args.roll_number.strip(),
        "parent_email": args.parent_email.strip(),
        "parent_password": args.parent_password,
        "embedding": embedding.tolist(),
    }

    print("Uploading student record and embedding to Render...", flush=True)
    result = _request_json(
        "POST",
        f"{api_base_url}/register_student",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    print(result.get("message", "Student registered successfully."), flush=True)
    print(f"Student ID: {result.get('student_id')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a ParentPing student by extracting the face embedding locally and uploading only the embedding to Render."
    )
    parser.add_argument("--api", default="https://parentping-api.onrender.com", help="ParentPing backend API URL.")
    parser.add_argument("--weights", default="best_resnet18_arcface_parentping.pth", help="Path to model weights.")
    parser.add_argument("--video", required=True, help="Path to student multi-angle face video.")
    parser.add_argument("--name", required=True, help="Student name.")
    parser.add_argument("--roll-number", required=True, help="Student roll number.")
    parser.add_argument("--parent-email", required=True, help="Parent email.")
    parser.add_argument("--parent-password", required=True, help="Password for parent portal login.")
    parser.add_argument("--admin-email", default="", help="Admin email. If omitted, prompted interactively.")
    parser.add_argument("--admin-password", default="", help="Admin password. If omitted, prompted securely.")
    parser.add_argument("--min-faces", type=int, default=5, help="Minimum clear face crops required from video.")
    parser.add_argument("--max-embeddings", type=int, default=20, help="Maximum face embeddings to average.")
    args = parser.parse_args()

    try:
        register_student(args)
    except Exception as exc:
        print(f"Registration failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

