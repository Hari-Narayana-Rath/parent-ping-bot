from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int, float]


class FaceDetector:
    def __init__(self, use_retinaface: bool = False, min_confidence: float = 0.5) -> None:
        self.use_retinaface = use_retinaface
        self.min_confidence = min_confidence
        self._retinaface = None

        if use_retinaface:
            try:
                from retinaface import RetinaFace  # type: ignore

                self._retinaface = RetinaFace
            except Exception:
                self._retinaface = None

        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, frame: np.ndarray) -> List[BBox]:
        if self._retinaface is not None:
            return self._detect_with_retinaface(frame)
        return self._detect_with_haar(frame)

    def _detect_with_haar(self, frame: np.ndarray) -> List[BBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        detections: List[BBox] = []
        for (x, y, w, h) in faces:
            detections.append((int(x), int(y), int(x + w), int(y + h), 1.0))
        return detections

    def _detect_with_retinaface(self, frame: np.ndarray) -> List[BBox]:
        results = self._retinaface.detect_faces(frame)  # type: ignore[union-attr]
        detections: List[BBox] = []
        if not isinstance(results, dict):
            return detections

        for value in results.values():
            score = float(value.get("score", 0.0))
            if score < self.min_confidence:
                continue
            x1, y1, x2, y2 = value["facial_area"]
            detections.append((int(x1), int(y1), int(x2), int(y2), score))
        return detections

    @staticmethod
    def crop_largest_face(frame: np.ndarray, detections: List[BBox]) -> Optional[np.ndarray]:
        if not detections:
            return None
        largest = max(detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        x1, y1, x2, y2, _ = largest
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2]

