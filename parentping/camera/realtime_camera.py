from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
        api_base_url: str = "http://127.0.0.1:8000",
        threshold: float = 0.125,
        use_retinaface: bool = False,
    ) -> None:
        model, device = load_embedding_model(model_weights_path)
        self.detector = FaceDetector(use_retinaface=use_retinaface)
        self.extractor = EmbeddingExtractor(model=model, device=device)
        self.matcher = SimilarityMatcher(threshold=threshold)
        self.validator = MultiFrameValidator(required_votes=3, window_size=5)
        self.db_path = str(db_path)
        self.api_base_url = api_base_url.rstrip("/")
        self.last_marked_time: Dict[int, float] = {}

    def _load_reference_embeddings(self) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
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

    def _mark_attendance_api(self, student_id: int) -> None:
        payload = json.dumps({"student_id": student_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base_url}/mark_attendance",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3):
                return
        except Exception:
            return

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
                predicted_for_frame: Optional[int] = None
                display_text = "Unknown"

                if detections:
                    x1, y1, x2, y2, _ = max(detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                    face = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]
                    if face.size > 0:
                        result = self._recognize_face(face, references)
                        predicted_for_frame = result.student_id
                        if result.student_id is not None:
                            display_text = f"{names.get(result.student_id, 'Student')} ({result.score:.3f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                confirmed_id = self.validator.add_prediction(predicted_for_frame)
                if confirmed_id is not None:
                    now = time.time()
                    last = self.last_marked_time.get(confirmed_id, 0.0)
                    if now - last > 30:
                        self._mark_attendance_api(confirmed_id)
                        self.last_marked_time[confirmed_id] = now

                cv2.putText(
                    frame,
                    display_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
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
        default="http://127.0.0.1:8000",
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
    args = parser.parse_args()

    service = RealtimeCameraService(
        model_weights_path=args.weights,
        db_path=args.db,
        api_base_url=args.api,
        threshold=args.threshold,
        use_retinaface=args.retinaface,
    )
    service.run()
