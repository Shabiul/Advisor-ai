"""
Temporal Buffer
===============
Sliding-window buffer that stores timestamped signal snapshots.
Uses collections.deque for efficient eviction.
"""

import time
from collections import deque


class TemporalBuffer:
    """Fixed-duration sliding window of signal snapshots."""

    def __init__(self, window_seconds: float = 10.0):
        self.window_seconds = window_seconds
        self._buffer: deque = deque()

    # ── public API ────────────────────────────────────────────────────

    def push(self, signals: dict):
        """Add a signal snapshot with the current timestamp."""
        entry = {
            "timestamp": signals.get("timestamp", time.time()),
            "signals": signals,
        }
        self._buffer.append(entry)
        self._evict()

    def get_window(self) -> list[dict]:
        """Return all entries in the current window (list of dicts)."""
        self._evict()
        return list(self._buffer)

    def clear(self):
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def duration(self) -> float:
        """Actual timespan covered by current buffer contents."""
        if len(self._buffer) < 2:
            return 0.0
        return self._buffer[-1]["timestamp"] - self._buffer[0]["timestamp"]

    # ── private ───────────────────────────────────────────────────────

    def _evict(self):
        """Remove entries older than the window."""
        if not self._buffer:
            return
        cutoff = time.time() - self.window_seconds
        while self._buffer and self._buffer[0]["timestamp"] < cutoff:
            self._buffer.popleft()
