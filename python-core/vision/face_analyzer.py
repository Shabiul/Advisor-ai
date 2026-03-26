"""
Face Analyzer Module
====================
Extracts facial signals from a video frame using MediaPipe Face Mesh.

Output schema (per frame):
{
  "gaze": "LEFT | RIGHT | CENTER",
  "eye_contact": 0.0-1.0,
  "blink_rate": <number>,          # blinks per minute (rolling)
  "smile": "NONE | SUBTLE | SOCIAL | GENUINE",
  "lips": "COMPRESSED | RELAXED | SPEAKING",
  "cheeks": "RAISED | RELAXED"
}
"""

import time
import math
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


# ── Landmark index constants ──────────────────────────────────────────────

# Iris / gaze
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Eyelid (upper / lower) – used for blink detection via EAR
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Mouth
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP_TOP = 0
LOWER_LIP_BOTTOM = 17

# Cheek raise (approximate via cheek-bone landmarks vs lower cheek)
LEFT_CHEEK_TOP = 123
LEFT_CHEEK_BOTTOM = 187
RIGHT_CHEEK_TOP = 352
RIGHT_CHEEK_BOTTOM = 411


def _distance(p1, p2):
    """Euclidean distance between two landmark-like objects (x, y)."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _iris_center(landmarks, indices):
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


class FaceAnalyzer:
    """Stateful face analyzer – keeps a blink counter for blink-rate."""

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Blink tracking
        self._blink_timestamps: list[float] = []
        self._eye_closed = False
        self._blink_ear_threshold = 0.21
        self._blink_window = 60  # seconds for rolling blink-rate

    # ── public API ────────────────────────────────────────────────────

    def analyze(self, frame_rgb) -> dict | None:
        """Return facial-signal dict or None if no face detected."""
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        gaze = self._compute_gaze(lm)
        eye_contact = self._compute_eye_contact(gaze)
        blink_rate = self._compute_blink_rate(lm)
        smile = self._compute_smile(lm)
        lips = self._compute_lips(lm)
        cheeks = self._compute_cheeks(lm)

        return {
            "gaze": gaze,
            "eye_contact": round(eye_contact, 2),
            "blink_rate": blink_rate,
            "smile": smile,
            "lips": lips,
            "cheeks": cheeks,
        }

    # ── private helpers ───────────────────────────────────────────────

    def _compute_gaze(self, lm) -> str:
        """Determine gaze direction from iris position relative to eye corners."""
        left_iris_x, _ = _iris_center(lm, LEFT_IRIS)
        left_inner_x = lm[LEFT_EYE_INNER].x
        left_outer_x = lm[LEFT_EYE_OUTER].x
        left_ratio = (left_iris_x - left_outer_x) / (left_inner_x - left_outer_x + 1e-6)

        right_iris_x, _ = _iris_center(lm, RIGHT_IRIS)
        right_inner_x = lm[RIGHT_EYE_INNER].x
        right_outer_x = lm[RIGHT_EYE_OUTER].x
        right_ratio = (right_iris_x - right_inner_x) / (right_outer_x - right_inner_x + 1e-6)

        avg = (left_ratio + right_ratio) / 2

        if avg < 0.35:
            return "RIGHT"
        elif avg > 0.65:
            return "LEFT"
        return "CENTER"

    @staticmethod
    def _compute_eye_contact(gaze: str) -> float:
        return 1.0 if gaze == "CENTER" else 0.0

    def _compute_blink_rate(self, lm) -> int:
        """Rolling blink-rate (blinks per minute)."""
        left_ear = _distance(lm[LEFT_EYE_TOP], lm[LEFT_EYE_BOTTOM]) / (
            _distance(lm[LEFT_EYE_INNER], lm[LEFT_EYE_OUTER]) + 1e-6
        )
        right_ear = _distance(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / (
            _distance(lm[RIGHT_EYE_INNER], lm[RIGHT_EYE_OUTER]) + 1e-6
        )
        ear = (left_ear + right_ear) / 2

        now = time.time()
        if ear < self._blink_ear_threshold:
            if not self._eye_closed:
                self._eye_closed = True
                self._blink_timestamps.append(now)
        else:
            self._eye_closed = False

        # Prune old blinks
        cutoff = now - self._blink_window
        self._blink_timestamps = [t for t in self._blink_timestamps if t > cutoff]

        return len(self._blink_timestamps)

    @staticmethod
    def _compute_smile(lm) -> str:
        """Classify smile intensity."""
        mouth_width = _distance(lm[LEFT_MOUTH], lm[RIGHT_MOUTH])
        mouth_height = _distance(lm[UPPER_LIP], lm[LOWER_LIP])
        ratio = mouth_height / (mouth_width + 1e-6)

        cheek_raise = (
            _distance(lm[LEFT_CHEEK_TOP], lm[LEFT_CHEEK_BOTTOM])
            + _distance(lm[RIGHT_CHEEK_TOP], lm[RIGHT_CHEEK_BOTTOM])
        ) / 2

        if ratio < 0.15:
            return "NONE"
        elif ratio < 0.30:
            return "SUBTLE"
        elif cheek_raise < 0.04:
            return "SOCIAL"
        return "GENUINE"

    @staticmethod
    def _compute_lips(lm) -> str:
        """Determine lip state."""
        lip_gap = _distance(lm[UPPER_LIP], lm[LOWER_LIP])
        mouth_width = _distance(lm[LEFT_MOUTH], lm[RIGHT_MOUTH])
        ratio = lip_gap / (mouth_width + 1e-6)

        if ratio > 0.25:
            return "SPEAKING"
        elif ratio < 0.05:
            return "COMPRESSED"
        return "RELAXED"

    @staticmethod
    def _compute_cheeks(lm) -> str:
        """Check for cheek raise (Duchenne indicator)."""
        left_raise = _distance(lm[LEFT_CHEEK_TOP], lm[LEFT_CHEEK_BOTTOM])
        right_raise = _distance(lm[RIGHT_CHEEK_TOP], lm[RIGHT_CHEEK_BOTTOM])
        avg = (left_raise + right_raise) / 2
        return "RAISED" if avg < 0.045 else "RELAXED"
