"""
Aggregation Engine (CORE)
=========================
Summarises the temporal buffer into cumulative ratios, event counts, and
event durations – all normalized by total frames / time.

Output schema:
{
  "off_screen_ratio": 0.32,
  "head_down_ratio": 0.18,
  "face_missing_ratio": 0.05,
  "slouch_ratio": 0.12,
  "event_counts": { "LOOK_AWAY": 12, ... },
  "event_durations": { "LOOK_AWAY": 40.0, ... }   # seconds
}
"""

from analytics.event_engine import EventEngine


class AggregationEngine:
    """Processes the full temporal buffer and produces cumulative metrics."""

    def __init__(self):
        self.event_engine = EventEngine()

    def aggregate(self, window: list[dict]) -> dict:
        """
        Parameters
        ----------
        window : list[dict]
            Output of TemporalBuffer.get_window().
            Each item: {"timestamp": float, "signals": {...}}

        Returns
        -------
        dict  with ratios, counts, and durations.
        """
        total = len(window)
        if total == 0:
            return self._empty()

        # Accumulate events across entire window
        event_counts: dict[str, int] = {}
        off_screen = 0
        head_down = 0
        face_missing = 0
        slouch = 0

        all_events: list[dict] = []

        for entry in window:
            signals = entry["signals"]
            events = self.event_engine.detect(signals)
            all_events.extend(events)

            event_types = {e["type"] for e in events}

            if "LOOK_AWAY" in event_types:
                off_screen += 1
            if "LOOK_DOWN" in event_types:
                head_down += 1
            if "FACE_MISSING" in event_types:
                face_missing += 1
            if "SLOUCH" in event_types:
                slouch += 1

            for e in events:
                event_counts[e["type"]] = event_counts.get(e["type"], 0) + 1

        # Duration estimation (approximate frame duration from timestamps)
        if total >= 2:
            total_time = window[-1]["timestamp"] - window[0]["timestamp"]
        else:
            total_time = 0.0

        frame_duration = total_time / max(total - 1, 1)

        event_durations: dict[str, float] = {}
        for etype, count in event_counts.items():
            event_durations[etype] = round(count * frame_duration, 2)

        return {
            "off_screen_ratio": round(off_screen / total, 2),
            "head_down_ratio": round(head_down / total, 2),
            "face_missing_ratio": round(face_missing / total, 2),
            "slouch_ratio": round(slouch / total, 2),
            "event_counts": event_counts,
            "event_durations": event_durations,
        }

    @staticmethod
    def _empty() -> dict:
        return {
            "off_screen_ratio": 0.0,
            "head_down_ratio": 0.0,
            "face_missing_ratio": 0.0,
            "slouch_ratio": 0.0,
            "event_counts": {},
            "event_durations": {},
        }
