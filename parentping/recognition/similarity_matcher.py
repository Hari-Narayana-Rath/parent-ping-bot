from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return -1.0
    return float(np.dot(a, b) / (a_norm * b_norm))


class SimilarityMatcher:
    def __init__(self, threshold: float = 0.125) -> None:
        self.threshold = threshold

    def match(
        self,
        query_embedding: np.ndarray,
        reference_embeddings: Dict[int, np.ndarray],
    ) -> Tuple[Optional[int], float]:
        best_student_id: Optional[int] = None
        best_score = -1.0

        for student_id, ref_embedding in reference_embeddings.items():
            score = cosine_similarity(query_embedding, ref_embedding)
            if score > best_score:
                best_score = score
                best_student_id = student_id

        if best_score >= self.threshold:
            return best_student_id, best_score
        return None, best_score


class MultiFrameValidator:
    """Confirms identity when at least required_votes in a sliding window agree."""

    def __init__(self, required_votes: int = 3, window_size: int = 5) -> None:
        self.required_votes = required_votes
        self.window_size = window_size
        self.predictions: Deque[Optional[int]] = deque(maxlen=window_size)

    def add_prediction(self, predicted_student_id: Optional[int]) -> Optional[int]:
        self.predictions.append(predicted_student_id)
        if len(self.predictions) < self.window_size:
            return None

        votes = Counter(p for p in self.predictions if p is not None)
        if not votes:
            return None

        winner, winner_count = votes.most_common(1)[0]
        if winner_count >= self.required_votes:
            self.predictions.clear()
            return winner
        return None

