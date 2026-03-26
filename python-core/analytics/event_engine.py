"""
Event Engine
============
Converts per-frame signals into discrete behavioral events.

Rules (from PRD):
  gaze != CENTER         → LOOK_AWAY
  neck == DOWN           → LOOK_DOWN
  face == None           → FACE_MISSING
  sitting_posture == SLOUCHED → SLOUCH
"""


class EventEngine:
    """Stateless event detector – evaluates one signal snapshot at a time."""

    def detect(self, signals: dict) -> list[dict]:
        """Return a list of event dicts for a single frame's signals."""
        events: list[dict] = []
        ts = signals.get("timestamp", 0)
        face = signals.get("face")
        pose = signals.get("pose")

        # ── Face events ───────────────────────────────────────────────
        if face is None:
            events.append({"type": "FACE_MISSING", "timestamp": ts})
        else:
            if face.get("gaze") != "CENTER":
                events.append({"type": "LOOK_AWAY", "timestamp": ts})

        # ── Pose events ───────────────────────────────────────────────
        if pose is not None:
            if pose.get("neck") == "DOWN":
                events.append({"type": "LOOK_DOWN", "timestamp": ts})

            if pose.get("sitting_posture") == "SLOUCHED":
                events.append({"type": "SLOUCH", "timestamp": ts})

            shoulders = pose.get("shoulders", {})
            if shoulders.get("energy") == "DROPPED":
                events.append({"type": "SHOULDER_DROP", "timestamp": ts})

            if pose.get("arms") == "CROSSED":
                events.append({"type": "ARMS_CROSSED", "timestamp": ts})

        return events
