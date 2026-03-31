"""
=============================================================
  Trusted Advisor AI — Session Recorder
  Automatically records video + behavioral signals when
  the camera pipeline is active. Records raw frames (no overlay)
  for clean training data, plus a JSONL signal log.
=============================================================
"""

import os
import cv2
import json
import time
import threading
import numpy as np
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that standard json can't serialize."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 2)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SessionRecorder:
    """
    Records webcam sessions with synchronized behavioral signal logs.

    Usage:
        recorder = SessionRecorder(base_dir="recordings", fps=30)
        recorder.start(frame_width=1280, frame_height=720)

        # In your frame loop:
        recorder.write_frame(raw_frame)
        recorder.write_signals(sig, report)

        # On exit:
        recorder.stop()

    Output structure:
        recordings/
          session_2026-03-31_13-45-00/
            video.mp4              ← raw webcam recording
            signals.jsonl          ← one JSON object per line
            metadata.json          ← session info
    """

    def __init__(self, base_dir=None, fps=30):
        """
        Args:
            base_dir: Directory to store recordings. Defaults to
                      <project_root>/recordings/
            fps: Target frames per second for video recording.
        """
        if base_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.join(project_root, "recordings")

        self._base_dir = base_dir
        self._fps = fps
        self._video_writer = None
        self._signals_file = None
        self._session_dir = None
        self._session_start = None
        self._frame_count = 0
        self._signal_count = 0
        self._running = False
        self._lock = threading.Lock()

    def start(self, frame_width=1280, frame_height=720):
        """
        Initialize recording for a new session.

        Args:
            frame_width: Width of video frames.
            frame_height: Height of video frames.
        """
        if self._running:
            return

        # Create session directory (with millisecond precision to avoid collisions)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000:03d}"
        self._session_dir = os.path.join(self._base_dir, f"session_{timestamp}")
        os.makedirs(self._session_dir, exist_ok=True)

        # Initialize video writer — try multiple codecs for compatibility
        video_path = os.path.join(self._session_dir, "video.mp4")
        codec_attempts = [
            ("mp4v", "video.mp4"),
            ("avc1", "video.mp4"),
            ("XVID", "video.avi"),
            ("MJPG", "video.avi"),
        ]

        self._video_writer = None
        for codec, fname in codec_attempts:
            video_path = os.path.join(self._session_dir, fname)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                video_path, fourcc, self._fps,
                (int(frame_width), int(frame_height)),
                isColor=True,
            )
            if writer.isOpened():
                self._video_writer = writer
                break
            else:
                writer.release()

        if self._video_writer is None:
            print("[RECORDER] WARNING: No video codec worked. Recording signals only.")

        # Initialize signals log (JSONL)
        signals_path = os.path.join(self._session_dir, "signals.jsonl")
        self._signals_file = open(signals_path, "w", encoding="utf-8")

        self._session_start = time.time()
        self._frame_count = 0
        self._signal_count = 0
        self._running = True

        print(f"[RECORDER] Session recording started -> {self._session_dir}")
        print(f"[RECORDER] Video: {video_path}")
        print(f"[RECORDER] Signals: {signals_path}")

    def write_frame(self, frame):
        """
        Write a single raw frame to the video file.
        Thread-safe. Call this for every captured frame.

        Args:
            frame: BGR numpy array (raw, pre-overlay frame).
        """
        if not self._running or self._video_writer is None:
            return
        if frame is None or not hasattr(frame, 'shape') or len(frame.shape) < 2:
            return

        with self._lock:
            try:
                self._video_writer.write(frame)
                self._frame_count += 1
            except Exception as e:
                print(f"[RECORDER] Frame write error: {e}")

    def write_signals(self, sig, report=None):
        """
        Write a signal snapshot to the JSONL log.
        Each line is a self-contained JSON object with timestamp.

        Args:
            sig: Current signal dictionary from the pipeline.
            report: Optional cumulative report dictionary.
        """
        if not self._running or self._signals_file is None:
            return

        with self._lock:
            try:
                entry = {
                    "timestamp": time.time(),
                    "elapsed_sec": round(time.time() - self._session_start, 3),
                    "frame_number": self._frame_count,
                    "signals": _safe_copy(sig),
                }
                if report:
                    entry["report"] = _safe_copy(report)

                line = json.dumps(entry, cls=NumpyEncoder, separators=(",", ":"))
                self._signals_file.write(line + "\n")
                self._signal_count += 1

                # Flush periodically to avoid data loss on crash
                if self._signal_count % 30 == 0:
                    self._signals_file.flush()
            except Exception as e:
                print(f"[RECORDER] Signal write error: {e}")

    def stop(self):
        """
        Finalize and close the recording session.
        Writes session metadata and releases all resources.
        """
        if not self._running:
            return

        self._running = False
        session_end = time.time()
        duration = round(session_end - self._session_start, 2) if self._session_start else 0

        with self._lock:
            # Release video writer
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None

            # Close signals file
            if self._signals_file is not None:
                self._signals_file.flush()
                self._signals_file.close()
                self._signals_file = None

        # Write session metadata
        if self._session_dir:
            metadata = {
                "session_id": os.path.basename(self._session_dir),
                "start_time": datetime.fromtimestamp(self._session_start).isoformat(),
                "end_time": datetime.fromtimestamp(session_end).isoformat(),
                "duration_seconds": duration,
                "total_frames": self._frame_count,
                "total_signals": self._signal_count,
                "fps": self._fps,
                "created_at": datetime.now().isoformat(),
            }
            metadata_path = os.path.join(self._session_dir, "metadata.json")
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"[RECORDER] Metadata write error: {e}")

        print(f"[RECORDER] Session recording stopped.")
        print(f"[RECORDER] Recorded {self._frame_count} frames, {self._signal_count} signal entries")
        print(f"[RECORDER] Duration: {duration:.1f}s -> {self._session_dir}")

    @property
    def is_recording(self):
        """Check if recording is active."""
        return self._running

    @property
    def session_dir(self):
        """Get current session directory path."""
        return self._session_dir

    @property
    def frame_count(self):
        """Get number of frames recorded so far."""
        return self._frame_count


def _safe_copy(obj):
    """Create a JSON-serializable copy of a dict, handling numpy types."""
    if isinstance(obj, dict):
        return {k: _safe_copy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_copy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return round(float(obj), 4)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
