"""
Camera Pipeline
===============
Captures webcam frames and routes them through Face and Pose analyzers.
Yields (frame, signals) tuples per frame for overlay + analytics.
"""

import time
import cv2

from .face_analyzer import FaceAnalyzer
from .pose_analyzer import PoseAnalyzer


class CameraPipeline:
    """Manages video capture and per-frame signal extraction."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.face_analyzer = FaceAnalyzer()
        self.pose_analyzer = PoseAnalyzer()
        self._cap = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def open(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── generator ─────────────────────────────────────────────────────

    def stream(self):
        """Yield (frame_bgr, signals_dict) continuously."""
        if self._cap is None:
            self.open()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_signals = self.face_analyzer.analyze(rgb)
            pose_signals = self.pose_analyzer.analyze(rgb)

            signals = {
                "timestamp": time.time(),
                "face": face_signals,   # None when no face
                "pose": pose_signals,   # None when body not visible
            }

            yield frame, signals

    # ── context manager ───────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
