"""
Event Tracker
=============
Tracks live behavioral events with precise timestamps and durations.
Maintains a history of look-away episodes and other events for the dashboard.

Each tracked event has:
  - type: LOOK_AWAY, LOOK_DOWN, FACE_MISSING, SLOUCH
  - start_time: ISO timestamp when event began
  - end_time: ISO timestamp when event ended (None if ongoing)
  - duration: seconds the event lasted
"""

import time
from datetime import datetime
from collections import deque


class EventTracker:
    """Tracks behavioral events with timestamps and durations."""

    MAX_HISTORY = 50  # Keep last 50 events for dashboard

    def __init__(self):
        # Active (ongoing) events keyed by type
        self._active: dict[str, dict] = {}
        # Completed event history
        self._history: deque = deque(maxlen=self.MAX_HISTORY)

    def update(self, current_events: list[dict]):
        """
        Update tracking with events detected in the current frame.

        Parameters
        ----------
        current_events : list[dict]
            Output from EventEngine.detect() — list of {"type": ..., "timestamp": ...}
        """
        now = time.time()
        current_types = {e["type"] for e in current_events}

        # Check which active events have ended (no longer in current frame)
        ended = [etype for etype in self._active if etype not in current_types]
        for etype in ended:
            event = self._active.pop(etype)
            event["end_time"] = now
            event["end_iso"] = datetime.fromtimestamp(now).strftime("%H:%M:%S")
            event["duration"] = round(now - event["start_time"], 2)
            self._history.append(event)

        # Start tracking new events
        for etype in current_types:
            if etype not in self._active:
                self._active[etype] = {
                    "type": etype,
                    "start_time": now,
                    "start_iso": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                    "end_time": None,
                    "end_iso": None,
                    "duration": 0,
                }

        # Update duration on active events
        for etype, event in self._active.items():
            event["duration"] = round(now - event["start_time"], 2)

    def get_active_events(self) -> list[dict]:
        """Return currently active (ongoing) events."""
        return list(self._active.values())

    def get_history(self) -> list[dict]:
        """Return completed events, most recent first."""
        return list(reversed(self._history))

    def get_live_alerts(self) -> list[dict]:
        """
        Return alerts for the dashboard — both active + recent history.
        Active events are marked with 'active': True.
        """
        alerts = []

        # Active events first
        for event in self._active.values():
            alerts.append({
                "type": event["type"],
                "start": event["start_iso"],
                "end": "ONGOING",
                "duration": f"{event['duration']:.1f}s",
                "active": True,
            })

        # Recent completed events
        for event in reversed(self._history):
            alerts.append({
                "type": event["type"],
                "start": event["start_iso"],
                "end": event["end_iso"],
                "duration": f"{event['duration']:.1f}s",
                "active": False,
            })

        return alerts[:30]  # Cap at 30 for dashboard

    def get_look_away_episodes(self) -> list[dict]:
        """
        Extract specifically LOOK_AWAY episodes with full details.
        Returns both active and completed look-away events.
        """
        episodes = []

        # Check active
        if "LOOK_AWAY" in self._active:
            ev = self._active["LOOK_AWAY"]
            episodes.append({
                "start": ev["start_iso"],
                "end": "ONGOING",
                "duration": f"{ev['duration']:.1f}s",
                "active": True,
            })

        # Completed look-away episodes
        for event in reversed(self._history):
            if event["type"] == "LOOK_AWAY":
                episodes.append({
                    "start": event["start_iso"],
                    "end": event["end_iso"],
                    "duration": f"{event['duration']:.1f}s",
                    "active": False,
                })

        return episodes
