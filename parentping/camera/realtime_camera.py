from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from parentping.models.embedding_model import load_embedding_model
from parentping.recognition.embedding_extractor import EmbeddingExtractor
from parentping.recognition.face_detector import FaceDetector
from parentping.recognition.similarity_matcher import MultiFrameValidator, SimilarityMatcher


@dataclass
class RecognitionResult:
    student_id: Optional[int]
    score: float


class RealtimeCameraService:
    def __init__(
        self,
        model_weights_path: str | Path = "best_resnet18_arcface_parentping.pth",
        db_path: str | Path = "parentping.db",
        api_base_url: str = "https://parentping-api.onrender.com",
        threshold: float = 0.125,
        use_retinaface: bool = False,
        max_faces: int = 3,
        presence_interval_sec: float = 0.45,
        camera_secret: str = "",
    ) -> None:
        model, device = load_embedding_model(model_weights_path)
        self.detector = FaceDetector(use_retinaface=use_retinaface)
        self.extractor = EmbeddingExtractor(model=model, device=device)
        self.matcher = SimilarityMatcher(threshold=threshold)
        self.validator = MultiFrameValidator(required_votes=3, window_size=5)
        self.db_path = str(db_path)
        self.api_base_url = api_base_url.rstrip("/")
        self.last_marked_time: Dict[int, float] = {}
        self.max_faces = max(1, min(int(max_faces), 8))
        self.presence_interval_sec = max(0.2, float(presence_interval_sec))
        self.camera_secret = (camera_secret or os.getenv("PARENTPING_CAMERA_SECRET", "")).strip()
        self._last_presence_post = 0.0
        self._warned_no_secret = False
        self._pending_attendance_posts: set[int] = set()
        self._last_attendance_error = 0.0
        self._last_presence_error = 0.0

    def _load_reference_embeddings_from_db(self) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, name, embedding_vector FROM students")
            rows = cur.fetchall()
        finally:
            conn.close()

        embeddings: Dict[int, np.ndarray] = {}
        names: Dict[int, str] = {}
        for student_id, name, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size == 512:
                embeddings[int(student_id)] = vec
                names[int(student_id)] = str(name)
        return embeddings, names

    def _load_reference_embeddings_from_api(self) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        if not self.camera_secret:
            raise RuntimeError("camera secret is required for backend embedding sync")
        req = urllib.request.Request(
            f"{self.api_base_url}/camera/students",
            headers={"X-Camera-Secret": self.camera_secret},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        embeddings: Dict[int, np.ndarray] = {}
        names: Dict[int, str] = {}
        if not isinstance(payload, list):
            raise RuntimeError("camera/students returned an invalid payload")

        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                student_id = int(item["id"])
                name = str(item.get("name") or "Student")
                vec = np.asarray(item.get("embedding", []), dtype=np.float32)
            except Exception:
                continue
            if vec.size == 512:
                embeddings[student_id] = vec
                names[student_id] = name
        return embeddings, names

    def _load_reference_embeddings(self) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        errors: list[str] = []
        if self.camera_secret:
            try:
                embeddings, names = self._load_reference_embeddings_from_api()
                if embeddings:
                    print(f"[ParentPing] loaded {len(embeddings)} student embeddings from backend API", flush=True)
                    return embeddings, names
                errors.append("backend API returned no embeddings")
            except Exception as exc:
                errors.append(f"backend API sync failed: {exc}")
        else:
            errors.append("camera secret not provided; backend embedding sync skipped")

        try:
            embeddings, names = self._load_reference_embeddings_from_db()
            if embeddings:
                print(
                    f"[ParentPing] loaded {len(embeddings)} student embeddings from local DB. "
                    f"Backend sync was not used ({'; '.join(errors)}).",
                    flush=True,
                )
                return embeddings, names
            errors.append("local DB returned no embeddings")
        except Exception as exc:
            errors.append(f"local DB failed: {exc}")

        raise RuntimeError("No student embeddings available. " + " | ".join(errors))

    def _mark_attendance_api(self, student_id: int) -> None:
        try:
            payload = json.dumps({"student_id": student_id}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.api_base_url}/mark_attendance",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                if resp.status == 200:
                    print(f"[ParentPing] attendance synced for student ID {student_id}", flush=True)
                else:
                    print(f"[ParentPing] mark_attendance HTTP {resp.status}", flush=True)
        except Exception as exc:
            now = time.time()
            if now - self._last_attendance_error > 10:
                print(
                    f"[ParentPing] mark_attendance failed: {exc}. "
                    "If Render was sleeping, keep the camera running and it will retry on the next confirmed face.",
                    flush=True,
                )
                self._last_attendance_error = now
        finally:
            self._pending_attendance_posts.discard(student_id)

    def _queue_mark_attendance_api(self, student_id: int) -> None:
        if student_id in self._pending_attendance_posts:
            return
        self._pending_attendance_posts.add(student_id)
        thread = threading.Thread(target=self._mark_attendance_api, args=(student_id,), daemon=True)
        thread.start()

    def _post_presence_api(self, student_ids: List[int]) -> None:
        if not self.camera_secret:
            if not self._warned_no_secret:
                print(
                    "[ParentPing] Live parent portal updates need PARENTPING_CAMERA_SECRET "
                    "(same value on API and camera).",
                    flush=True,
                )
                self._warned_no_secret = True
            return
        body = json.dumps({"student_ids": student_ids}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base_url}/camera/presence",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Camera-Secret": self.camera_secret,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status != 200:
                    print(f"[ParentPing] camera/presence HTTP {resp.status}", flush=True)
        except Exception as exc:
            now = time.time()
            if now - self._last_presence_error > 10:
                print(f"[ParentPing] camera/presence failed: {exc}", flush=True)
                self._last_presence_error = now

    def _recognize_face(self, face_img: np.ndarray, references: Dict[int, np.ndarray]) -> RecognitionResult:
        embedding = self.extractor.extract(face_img)
        student_id, score = self.matcher.match(embedding, references)
        return RecognitionResult(student_id=student_id, score=score)

    def run(self) -> None:
        references, names = self._load_reference_embeddings()
        if not references:
            raise RuntimeError("No student embeddings available in database.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Webcam could not be opened.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                detections = self.detector.detect_faces(frame)
                primary_prediction: Optional[int] = None
                visible_ids: List[int] = []
                hud_lines: List[str] = []

                if detections:
                    sorted_dets = sorted(
                        detections,
                        key=lambda d: (d[2] - d[0]) * (d[3] - d[1]),
                        reverse=True,
                    )[: self.max_faces]

                    for idx, (x1, y1, x2, y2, _) in enumerate(sorted_dets):
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        face = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]
                        if face.size == 0:
                            continue

                        result = self._recognize_face(face, references)
                        if idx == 0:
                            primary_prediction = result.student_id

                        label = "Unknown"
                        color = (0, 165, 255)
                        if result.student_id is not None:
                            visible_ids.append(int(result.student_id))
                            label = f"{names.get(result.student_id, 'Student')} ({result.score:.2f})"
                            color = (0, 255, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                        hud_lines.append(label)

                now = time.time()
                if now - self._last_presence_post >= self.presence_interval_sec:
                    unique_visible = sorted({sid for sid in visible_ids if sid > 0})
                    self._post_presence_api(unique_visible)
                    self._last_presence_post = now

                confirmed_id = self.validator.add_prediction(primary_prediction)
                if confirmed_id is not None:
                    mark_now = time.time()
                    last = self.last_marked_time.get(confirmed_id, 0.0)
                    if mark_now - last > 30:
                        self._queue_mark_attendance_api(confirmed_id)
                        self.last_marked_time[confirmed_id] = mark_now

                status_hud = " | ".join(hud_lines) if hud_lines else "No face"
                cv2.putText(
                    frame,
                    status_hud[:120],
                    (10, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("ParentPing Realtime Attendance", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParentPing realtime camera attendance service")
    parser.add_argument(
        "--weights",
        default="best_resnet18_arcface_parentping.pth",
        help="Path to ArcFace inference weights (.pth).",
    )
    parser.add_argument(
        "--db",
        default="parentping.db",
        help="Path to SQLite DB file.",
    )
    parser.add_argument(
        "--api",
        default="https://parentping-api.onrender.com",
        help="FastAPI base URL for attendance marking.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.125,
        help="Cosine similarity threshold.",
    )
    parser.add_argument(
        "--retinaface",
        action="store_true",
        help="Use RetinaFace detector if installed.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=3,
        help="Recognize up to this many faces per frame (largest boxes first).",
    )
    parser.add_argument(
        "--presence-interval",
        type=float,
        default=0.45,
        help="Seconds between live presence POSTs to the API.",
    )
    parser.add_argument(
        "--camera-secret",
        default=os.getenv("PARENTPING_CAMERA_SECRET", ""),
        help="Must match API env PARENTPING_CAMERA_SECRET for live parent portal status.",
    )
    args = parser.parse_args()

    service = RealtimeCameraService(
        model_weights_path=args.weights,
        db_path=args.db,
        api_base_url=args.api,
        threshold=args.threshold,
        use_retinaface=args.retinaface,
        max_faces=args.max_faces,
        presence_interval_sec=args.presence_interval,
        camera_secret=args.camera_secret,
    )
    service.run()

