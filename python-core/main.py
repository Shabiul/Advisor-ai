"""
=============================================================
  Trusted Advisor AI — Main Pipeline (V6)
  Full vision → analytics → HTTP → dashboard pipeline
  Based on the working reference camera_pipeline.py
=============================================================
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import json
import time
import threading
import urllib.request
import urllib.error
from collections import deque, Counter
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CAMERA_INDEX = int(os.environ.get("ADVISOR_CAMERA", "0"))
BACKEND_URL = os.environ.get("ADVISOR_BACKEND_URL", "http://localhost:3000/analyze")
ADVISOR_MODE = os.environ.get("ADVISOR_MODE", "PROCTORING")
EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 2
BLINK_WINDOW_SEC = 60
SMOOTH_WINDOW = 5
REPORT_INTERVAL = 0.5  # seconds between HTTP posts

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473
NOSE_TIP_IDX = 1
L_EYE_C_IDX = 33
R_EYE_C_IDX = 263
FOREHEAD_IDX = 10
CHIN_IDX = 152

L_BROW_IDX = [336, 296, 334, 293, 300]
R_BROW_IDX = [107, 66, 105, 63, 70]
L_BROW_BASE = [362, 385]
R_BROW_BASE = [33, 160]

UPPER_LIP_IDX = [13, 312, 311, 310, 415, 308]
LOWER_LIP_IDX = [14, 82, 81, 80, 91, 78]
LIP_LEFT_IDX = 61
LIP_RIGHT_IDX = 291

L_CHEEK_IDX = [50, 101, 118, 36]
R_CHEEK_IDX = [280, 330, 347, 266]

# Pose indices
P_LEFT_SHOULDER = 11
P_RIGHT_SHOULDER = 12
P_LEFT_ELBOW = 13
P_RIGHT_ELBOW = 14
P_LEFT_WRIST = 15
P_RIGHT_WRIST = 16
P_LEFT_HIP = 23
P_RIGHT_HIP = 24
P_NOSE = 0


# ─────────────────────────────────────────────
#  NUMPY ENCODER
# ─────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
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


# ─────────────────────────────────────────────
#  TEMPORAL SMOOTHING
# ─────────────────────────────────────────────
def _mode_vote(history):
    if not history:
        return None
    recent = list(history)[-SMOOTH_WINDOW:]
    counts = Counter(recent)
    return counts.most_common(1)[0][0]


# ─────────────────────────────────────────────
#  SIGNAL FUNCTIONS
# ─────────────────────────────────────────────
def ear(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C else 0.0


def get_gaze(lms, w, h):
    l_inner, l_outer = lms[133].x, lms[33].x
    l_iris = lms[LEFT_IRIS_IDX].x
    l_eye_w = abs(l_outer - l_inner)
    l_ratio = (l_iris - min(l_inner, l_outer)) / l_eye_w if l_eye_w > 0.001 else 0.5

    r_inner, r_outer = lms[362].x, lms[263].x
    r_iris = lms[RIGHT_IRIS_IDX].x
    r_eye_w = abs(r_outer - r_inner)
    r_ratio = (r_iris - min(r_inner, r_outer)) / r_eye_w if r_eye_w > 0.001 else 0.5

    avg_ratio = (l_ratio + r_ratio) / 2.0
    if avg_ratio < 0.38:
        return "LOOKING LEFT"
    elif avg_ratio > 0.62:
        return "LOOKING RIGHT"
    return "CENTER"


def get_eye_contact(lms):
    l_inner, l_outer = lms[133].x, lms[33].x
    l_iris = lms[LEFT_IRIS_IDX].x
    l_eye_w = abs(l_outer - l_inner)
    l_ratio = (l_iris - min(l_inner, l_outer)) / l_eye_w if l_eye_w > 0.001 else 0.5

    r_inner, r_outer = lms[362].x, lms[263].x
    r_iris = lms[RIGHT_IRIS_IDX].x
    r_eye_w = abs(r_outer - r_inner)
    r_ratio = (r_iris - min(r_inner, r_outer)) / r_eye_w if r_eye_w > 0.001 else 0.5

    avg = (l_ratio + r_ratio) / 2.0
    dist = abs(avg - 0.5)
    return round(max(0.0, 1.0 - dist / 0.25), 2)


def get_head_pose(lms):
    nose = lms[NOSE_TIP_IDX]
    eye_mid_x = (lms[L_EYE_C_IDX].x + lms[R_EYE_C_IDX].x) / 2
    eye_mid_y = (lms[L_EYE_C_IDX].y + lms[R_EYE_C_IDX].y) / 2
    yaw = nose.x - eye_mid_x
    pitch = nose.y - eye_mid_y
    if abs(yaw) > 0.04:
        return yaw, pitch, "TURNED LEFT" if yaw < 0 else "TURNED RIGHT"
    if pitch > 0.25:
        return yaw, pitch, "LOOKING DOWN"
    if pitch < 0.15:
        return yaw, pitch, "HEAD RAISED"
    return yaw, pitch, "FACING FORWARD"


def get_brow(lms):
    face_h = abs(lms[FOREHEAD_IDX].y - lms[CHIN_IDX].y)
    if face_h < 0.001:
        return 0.5, "NEUTRAL"
    l_dist = (
        np.mean([lms[i].y for i in L_BROW_BASE])
        - np.mean([lms[i].y for i in L_BROW_IDX])
    ) / face_h
    r_dist = (
        np.mean([lms[i].y for i in R_BROW_BASE])
        - np.mean([lms[i].y for i in R_BROW_IDX])
    ) / face_h
    score = round(max(0.0, min((((l_dist + r_dist) / 2) - 0.02) / 0.10, 1.0)), 2)
    label = "RAISED" if score > 0.55 else "FURROWED" if score < 0.40 else "NEUTRAL"
    return score, label


def get_lip(lms, lip_hist=None):
    face_h = abs(lms[FOREHEAD_IDX].y - lms[CHIN_IDX].y)
    if face_h < 0.001:
        return {"state": "RELAXED", "movement": "LOW"}, 0.5

    lip_h = abs(
        np.mean([lms[i].y for i in LOWER_LIP_IDX])
        - np.mean([lms[i].y for i in UPPER_LIP_IDX])
    ) / face_h

    score = round(max(0.0, 1.0 - min(lip_h / 0.04, 1.0)), 2)

    if lip_h > 0.06:
        state = "SPEAKING"
    elif lip_h > 0.035:
        state = "SLIGHTLY_OPEN"
    elif score > 0.65:
        state = "COMPRESSED"
    elif score > 0.40:
        state = "TIGHT"
    else:
        state = "RELAXED"

    movement = "LOW"
    if lip_hist is not None:
        lip_hist.append(lip_h)
        if len(lip_hist) >= 3:
            recent = list(lip_hist)[-5:]
            variance = max(recent) - min(recent)
            if variance > 0.015:
                movement = "HIGH"
            elif variance > 0.006:
                movement = "MEDIUM"

    if movement == "HIGH" and lip_h > 0.025:
        state = "SPEAKING"

    return {"state": state, "movement": movement}, score


def get_cheek_state(lms):
    face_h = abs(lms[FOREHEAD_IDX].y - lms[CHIN_IDX].y)
    if face_h < 0.001:
        return "RELAXED"

    l_cheek_y = np.mean([lms[i].y for i in L_CHEEK_IDX])
    r_cheek_y = np.mean([lms[i].y for i in R_CHEEK_IDX])
    cheek_mid_y = (l_cheek_y + r_cheek_y) / 2.0

    avg_ear_val = (ear(lms, LEFT_EYE_IDX, 1, 1) + ear(lms, RIGHT_EYE_IDX, 1, 1)) / 2
    mouth_y = (lms[LIP_LEFT_IDX].y + lms[LIP_RIGHT_IDX].y) / 2.0
    cheek_mouth_gap = (mouth_y - cheek_mid_y) / face_h

    if cheek_mouth_gap > 0.12 and avg_ear_val < 0.28:
        return "RAISED"
    if cheek_mouth_gap < 0.08:
        return "TENSE"
    return "RELAXED"


def get_smile(lms, ear_hist=None):
    face_w = abs(lms[L_EYE_C_IDX].x - lms[R_EYE_C_IDX].x)
    if face_w < 0.001:
        return False, "NONE"

    mouth_w = abs(lms[LIP_LEFT_IDX].x - lms[LIP_RIGHT_IDX].x) / face_w
    avg_ear_val = (ear(lms, LEFT_EYE_IDX, 1, 1) + ear(lms, RIGHT_EYE_IDX, 1, 1)) / 2

    if ear_hist is not None:
        ear_hist.append(avg_ear_val)

    eye_relaxed = False
    if ear_hist is not None and len(ear_hist) >= 3:
        recent = list(ear_hist)[-5:]
        ear_variance = max(recent) - min(recent)
        ear_mean = sum(recent) / len(recent)
        eye_relaxed = ear_variance < 0.04 and ear_mean < 0.30

    if mouth_w > 1.2 and eye_relaxed:
        return True, "GENUINE"
    if mouth_w > 1.2:
        return False, "SOCIAL"
    if mouth_w > 1.05:
        return False, "SUBTLE"
    return False, "NONE"


def get_nod(pitch_hist):
    if len(pitch_hist) < 6:
        return False
    recent = list(pitch_hist)[-6:]
    return np.mean([abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]) > 0.008


def get_shake(yaw_hist):
    if len(yaw_hist) < 8:
        return False
    recent = list(yaw_hist)[-8:]
    changes = sum(
        1 for i in range(2, len(recent))
        if (recent[i] - recent[i - 1]) * (recent[i - 1] - recent[i - 2]) < 0
    )
    return changes >= 3


# ─────────────────────────────────────────────
#  POSE HELPERS
# ─────────────────────────────────────────────
def _dist(a, b):
    import math
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def detect_arm_state(plm):
    lw = plm[P_LEFT_WRIST]
    rw = plm[P_RIGHT_WRIST]
    le = plm[P_LEFT_ELBOW]
    re = plm[P_RIGHT_ELBOW]

    if _dist(lw, re) < 0.12 and _dist(rw, le) < 0.12:
        return "CROSSED"

    wrist_sep = abs(lw.x - rw.x)
    lw_vis = getattr(lw, "visibility", 1.0)
    rw_vis = getattr(rw, "visibility", 1.0)
    if wrist_sep > 0.35 and lw_vis > 0.5 and rw_vis > 0.5:
        return "OPEN"

    return "RELAXED"


def detect_shoulder_advanced(plm):
    ls = plm[P_LEFT_SHOULDER]
    rs = plm[P_RIGHT_SHOULDER]
    nose = plm[P_NOSE]
    lh = plm[P_LEFT_HIP]
    rh = plm[P_RIGHT_HIP]

    y_diff = abs(ls.y - rs.y)
    alignment = "TILTED" if y_diff > 0.04 else "STRAIGHT"

    shoulder_mid_y = (ls.y + rs.y) / 2.0
    hip_mid_y = (lh.y + rh.y) / 2.0
    torso_h = abs(hip_mid_y - shoulder_mid_y)
    hip_vis = getattr(lh, "visibility", 0)
    if hip_vis > 0.3 and torso_h < 0.03:
        energy = "DROPPED"
    else:
        energy = "ACTIVE"

    nose_below = nose.y - shoulder_mid_y
    if nose_below > 0.04:
        position = "FORWARD"
    else:
        position = "NEUTRAL"

    return {"alignment": alignment, "energy": energy, "position": position}


_neck_angle_hist = deque(maxlen=10)


def detect_neck_position(plm):
    nose = plm[P_NOSE]
    ls = plm[P_LEFT_SHOULDER]
    rs = plm[P_RIGHT_SHOULDER]

    shoulder_mid_x = (ls.x + rs.x) / 2.0
    shoulder_mid_y = (ls.y + rs.y) / 2.0

    vertical_gap = nose.y - shoulder_mid_y
    lateral_offset = nose.x - shoulder_mid_x

    _neck_angle_hist.append((lateral_offset, vertical_gap))

    if vertical_gap > 0.08:
        position = "DOWN"
    elif vertical_gap > 0.06:
        position = "FORWARD_HEAD"
    elif lateral_offset < -0.04:
        position = "TILTED_RIGHT"
    elif lateral_offset > 0.04:
        position = "TILTED_LEFT"
    else:
        position = "STRAIGHT"

    stability = "STABLE"
    if len(_neck_angle_hist) >= 5:
        recent = list(_neck_angle_hist)[-5:]
        x_vals = [p[0] for p in recent]
        y_vals = [p[1] for p in recent]
        x_var = max(x_vals) - min(x_vals)
        y_var = max(y_vals) - min(y_vals)
        if x_var > 0.02 or y_var > 0.02:
            stability = "UNSTABLE"

    return {"position": position, "stability": stability}


_prev_shoulder_pos = None


def detect_sitting_posture(plm):
    global _prev_shoulder_pos

    ls = plm[P_LEFT_SHOULDER]
    rs = plm[P_RIGHT_SHOULDER]
    nose = plm[P_NOSE]
    lh = plm[P_LEFT_HIP]
    rh = plm[P_RIGHT_HIP]

    shoulder_mid_x = (ls.x + rs.x) / 2.0
    shoulder_mid_y = (ls.y + rs.y) / 2.0
    hip_mid_y = (lh.y + rh.y) / 2.0

    current_pos = (shoulder_mid_x, shoulder_mid_y)
    if _prev_shoulder_pos is not None:
        dx = abs(current_pos[0] - _prev_shoulder_pos[0])
        dy = abs(current_pos[1] - _prev_shoulder_pos[1])
        if dx > 0.03 or dy > 0.03:
            _prev_shoulder_pos = current_pos
            return "SHIFTING_UNSTABLE"
    _prev_shoulder_pos = current_pos

    nose_forward = nose.y - shoulder_mid_y
    torso_h = abs(hip_mid_y - shoulder_mid_y)

    hip_vis = getattr(lh, "visibility", 0)
    if hip_vis > 0.3 and torso_h < 0.03 and nose_forward > 0.08:
        return "SLOUCHED"

    if nose_forward > 0.03:
        return "LEAN_FORWARD"
    if nose_forward < -0.02:
        return "LEAN_BACK"

    return "UPRIGHT"


def detect_attention_state(face_data, pose_data):
    score = 0
    max_score = 4

    ec = face_data.get("eye_contact_score", 0)
    if ec > 0.6:
        score += 1

    head = face_data.get("head_pose", "")
    if "FORWARD" in head:
        score += 1

    shoulders = pose_data.get("shoulders", {})
    sh_align = shoulders.get("alignment", "") if isinstance(shoulders, dict) else shoulders
    if sh_align == "STRAIGHT":
        score += 1

    arms = pose_data.get("arms", "")
    if arms == "OPEN":
        score += 1

    if score >= 3:
        state = "ATTENTIVE"
    elif score == 2:
        state = "NEUTRAL"
    else:
        state = "DISTRACTED"

    return state, round(score / max_score, 2)


# ─────────────────────────────────────────────
#  COMPOSITE SCORES
# ─────────────────────────────────────────────
def tension_score(brow_s, lip_s, bpm, emotion):
    s = 0
    if brow_s < 0.35:
        s += 3
    if lip_s > 0.60:
        s += 3
    if bpm > 0 and bpm > 25:
        s += 2
    if bpm > 0 and bpm < 5:
        s += 1
    if emotion in ["angry", "fear", "disgust"]:
        s += 2
    return min(s, 10)


def engagement_score_calc(sig):
    gaze = sig.get("gaze", "CENTER")
    bpm = sig.get("blinks_per_minute", 15)
    ec = sig.get("eye_contact_score", 0.5)
    ten = sig.get("micro_tension_score", 0)

    s = 5
    if gaze == "CENTER":
        s += 2
    if ec > 0.7:
        s += 1
    if sig.get("nodding"):
        s += 1
    if sig.get("smile_genuine"):
        s += 1
    if bpm > 0 and 8 <= bpm <= 20:
        s += 1
    s -= ten // 3
    return max(0, min(s, 10))


# ─────────────────────────────────────────────
#  EVENT TRACKER (live timestamps + durations)
# ─────────────────────────────────────────────
class EventTracker:
    MAX_HISTORY = 50

    def __init__(self):
        self._active = {}
        self._history = deque(maxlen=self.MAX_HISTORY)

    def update(self, sig):
        """Detect events from signals and track them."""
        now = time.time()
        current_types = set()

        # Detect events from signal state
        gaze = sig.get("gaze", "CENTER")
        if gaze != "CENTER":
            current_types.add("LOOK_AWAY")

        face_detected = sig.get("face_detected", True)
        if not face_detected:
            current_types.add("FACE_MISSING")

        head_pose = sig.get("head_pose", "")
        if "DOWN" in head_pose:
            current_types.add("LOOK_DOWN")

        bl = sig.get("body_language", {})
        sitting = bl.get("sitting_posture", "")
        if "SLOUCH" in sitting:
            current_types.add("SLOUCHED_POSTURE")

        neck = bl.get("neck", {})
        neck_pos = neck.get("position", "") if isinstance(neck, dict) else ""
        if neck_pos == "FORWARD_HEAD":
            current_types.add("FORWARD_HEAD")

        # End events no longer active
        ended = [etype for etype in self._active if etype not in current_types]
        for etype in ended:
            event = self._active.pop(etype)
            event["end_time"] = now
            event["end_iso"] = datetime.fromtimestamp(now).strftime("%H:%M:%S")
            event["duration"] = round(now - event["start_time"], 2)
            self._history.append(event)

        # Start new events
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

        # Update durations
        for event in self._active.values():
            event["duration"] = round(now - event["start_time"], 2)

    def get_active_events(self):
        return list(self._active.values())

    def get_live_alerts(self):
        alerts = []
        for event in self._active.values():
            alerts.append({
                "type": event["type"],
                "start": event["start_iso"],
                "end": "ONGOING",
                "duration": f"{event['duration']:.1f}s",
                "active": True,
            })
        for event in reversed(self._history):
            alerts.append({
                "type": event["type"],
                "start": event["start_iso"],
                "end": event["end_iso"],
                "duration": f"{event['duration']:.1f}s",
                "active": False,
            })
        return alerts[:30]

    def get_look_away_episodes(self):
        episodes = []
        if "LOOK_AWAY" in self._active:
            ev = self._active["LOOK_AWAY"]
            episodes.append({
                "start": ev["start_iso"],
                "end": "ONGOING",
                "duration": f"{ev['duration']:.1f}s",
                "active": True,
            })
        for event in reversed(self._history):
            if event["type"] == "LOOK_AWAY":
                episodes.append({
                    "start": event["start_iso"],
                    "end": event["end_iso"],
                    "duration": f"{event['duration']:.1f}s",
                    "active": False,
                })
        return episodes


# ─────────────────────────────────────────────
#  REPORT GENERATOR
# ─────────────────────────────────────────────
class CumulativeReporter:
    """Generates cumulative reports from sliding window of signals."""

    def __init__(self, window_sec=10.0):
        self._window = deque()
        self._window_sec = window_sec

    def push(self, sig):
        now = time.time()
        self._window.append({"ts": now, "sig": sig.copy()})
        # Evict old entries
        cutoff = now - self._window_sec
        while self._window and self._window[0]["ts"] < cutoff:
            self._window.popleft()

    def generate(self, sig, event_tracker):
        total = len(self._window)
        if total == 0:
            return self._empty(event_tracker)

        # Count events in window
        off_screen = 0
        head_down = 0
        face_missing = 0
        slouch = 0

        for entry in self._window:
            s = entry["sig"]
            if s.get("gaze", "CENTER") != "CENTER":
                off_screen += 1
            if "DOWN" in s.get("head_pose", ""):
                head_down += 1
            if not s.get("face_detected", True):
                face_missing += 1
            if "SLOUCH" in s.get("body_language", {}).get("sitting_posture", ""):
                slouch += 1

        off_ratio = off_screen / total
        head_ratio = head_down / total
        face_ratio = face_missing / total
        slouch_ratio = slouch / total

        # Attention score
        penalty = (
            0.35 * off_ratio +
            0.25 * head_ratio +
            0.25 * face_ratio +
            0.15 * slouch_ratio
        )
        attention_score = round(max(0.0, min(1.0, 1.0 - penalty)), 2)

        if attention_score >= 0.70:
            focus_level = "HIGH"
        elif attention_score >= 0.40:
            focus_level = "MEDIUM"
        else:
            focus_level = "LOW"

        bl = sig.get("body_language", {})
        neck = bl.get("neck", {})
        shoulders = bl.get("shoulders", {})

        report = {
            "summary": {
                "attention_score": attention_score,
                "focus_level": focus_level,
                "stability": neck.get("stability", "STABLE") if isinstance(neck, dict) else "STABLE",
            },
            "metrics": {
                "off_screen_time": f"{int(off_ratio * 100)}%",
                "head_down_time": f"{int(head_ratio * 100)}%",
                "face_missing_time": f"{int(face_ratio * 100)}%",
                "slouch_time": f"{int(slouch_ratio * 100)}%",
            },
            "alerts": [],
            "body_analysis": {
                "posture": bl.get("sitting_posture", "UNKNOWN"),
                "neck": neck.get("position", "UNKNOWN") if isinstance(neck, dict) else str(neck),
                "shoulders": shoulders.get("energy", "UNKNOWN") if isinstance(shoulders, dict) else str(shoulders),
            },
            "recommendation": "GOOD" if attention_score >= 0.70 else "ACCEPTABLE" if attention_score >= 0.40 else "REVIEW_REQUIRED",
            "live_alerts": event_tracker.get_live_alerts(),
            "look_away_episodes": event_tracker.get_look_away_episodes(),
            "engagement_score": sig.get("engagement_score", 5),
            "eye_contact_score": sig.get("eye_contact_score", 0),
        }

        return report

    def _empty(self, event_tracker):
        return {
            "summary": {"attention_score": 1.0, "focus_level": "HIGH", "stability": "STABLE"},
            "metrics": {"off_screen_time": "0%", "head_down_time": "0%", "face_missing_time": "0%", "slouch_time": "0%"},
            "alerts": [],
            "body_analysis": {"posture": "UNKNOWN", "neck": "UNKNOWN", "shoulders": "UNKNOWN"},
            "recommendation": "GOOD",
            "live_alerts": event_tracker.get_live_alerts(),
            "look_away_episodes": event_tracker.get_look_away_episodes(),
            "engagement_score": 5,
            "eye_contact_score": 0,
        }


# ─────────────────────────────────────────────
#  BACKEND POSTER
# ─────────────────────────────────────────────
class BackendPoster:
    def __init__(self, url, mode):
        self._url = url
        self._mode = mode
        self._pending = None
        self._lock = threading.Lock()
        self._running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        print(f"[BACKEND] Poster active → {url} (mode={mode})")

    def push(self, report):
        with self._lock:
            self._pending = report

    def _run(self):
        while self._running:
            report = None
            with self._lock:
                if self._pending is not None:
                    report = self._pending
                    self._pending = None

            if report is not None:
                try:
                    payload = json.dumps(
                        {"mode": self._mode, "data": report},
                        cls=NumpyEncoder,
                    ).encode("utf-8")
                    req = urllib.request.Request(
                        self._url,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=2)
                except (urllib.error.URLError, OSError):
                    pass
                except Exception as e:
                    print(f"[BACKEND] Post error: {e}")

            time.sleep(0.3)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────
#  DEBUG OVERLAY
# ─────────────────────────────────────────────
def draw_overlay(frame, sig, report, active_events):
    h, w = frame.shape[:2]

    # Left panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, h), (10, 18, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 25

    def put(t, col=(200, 200, 200), bold=False):
        nonlocal y
        cv2.putText(
            frame, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, col, 2 if bold else 1, cv2.LINE_AA,
        )
        y += 18

    put("TRUSTED ADVISOR AI", (80, 200, 255), bold=True)
    put("-" * 35, (40, 60, 80))

    put(f"GAZE:    {sig.get('gaze', '-')}", (0, 200, 180))
    put(f"POSE:    {sig.get('head_pose', '-')}", (0, 180, 220))
    put(f"EYE:     {sig.get('eye_contact_score', 0):.2f}", (180, 180, 180))
    put(f"BLINKS:  {sig.get('blinks_per_minute', 0):.0f}/min", (180, 180, 180))
    put("-" * 35, (40, 60, 80))

    bl = sig.get("body_language", {})
    put(f"ARMS:    {bl.get('arms', '-')}", (200, 200, 100))
    sh = bl.get("shoulders", {})
    if isinstance(sh, dict):
        put(f"SH-ALGN: {sh.get('alignment', '-')}", (200, 200, 100))
        put(f"SH-ENRG: {sh.get('energy', '-')}", (200, 200, 100))
    nk = bl.get("neck", {})
    if isinstance(nk, dict):
        nk_col = (0, 220, 100) if nk.get('position') == 'STRAIGHT' else (200, 165, 0)
        put(f"NECK:    {nk.get('position', '-')} [{nk.get('stability', '-')}]", nk_col)
    sit = bl.get("sitting_posture", "-")
    sit_col = (0, 220, 100) if "UPRIGHT" in sit else (200, 165, 0)
    put(f"SIT:     {sit}", sit_col)
    put("-" * 35, (40, 60, 80))

    # Attention from report
    summary = report.get("summary", {})
    score = summary.get("attention_score", 0)
    focus = summary.get("focus_level", "N/A")
    score_col = (0, 220, 100) if focus == "HIGH" else (0, 180, 255) if focus == "MEDIUM" else (0, 80, 255)
    put(f"ATTENTION: {score:.0%}  [{focus}]", score_col, bold=True)

    eng = sig.get("engagement_score", 0)
    ten = sig.get("micro_tension_score", 0)
    put(f"ENGAGE:  {eng}/10", (0, 220, 100) if eng >= 6 else (200, 165, 0))
    put(f"TENSION: {ten}/10", (0, 50, 255) if ten >= 6 else (0, 220, 100))
    put("-" * 35, (40, 60, 80))

    # Active alerts on overlay
    put("ACTIVE ALERTS:", (80, 200, 255), bold=True)
    if not active_events:
        put("  None", (100, 100, 100))
    else:
        for ev in active_events[:5]:
            put(f"  >> {ev['type']} ({ev['duration']:.1f}s)", (0, 80, 255))

    # Brow / Smile
    put("-" * 35, (40, 60, 80))
    put(f"BROW:    {sig.get('brow_label', '-')}", (180, 180, 180))
    put(f"SMILE:   {sig.get('smile_label', '-')}", (180, 180, 180))
    nod_str = "YES" if sig.get("nodding") else "NO"
    shake_str = "YES" if sig.get("head_shake") else "NO"
    put(f"NOD: {nod_str}  SHAKE: {shake_str}", (180, 180, 180))

    # Timestamp
    cv2.putText(
        frame, time.strftime("%H:%M:%S"),
        (w - 80, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1,
    )


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
        return

    print("[Trusted Advisor AI] Starting V6 pipeline...")
    print(f"  Camera     : {CAMERA_INDEX}")
    print(f"  Backend URL: {BACKEND_URL}")
    print(f"  Mode       : {ADVISOR_MODE}")
    print("  Press 'q' in the camera window to quit.\n")

    # Init models
    fmesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    pose_model = mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    hands_model = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    # Init trackers
    event_tracker = EventTracker()
    reporter = CumulativeReporter(window_sec=10.0)
    backend_poster = BackendPoster(BACKEND_URL, ADVISOR_MODE)

    # Histories
    pitch_hist = deque(maxlen=20)
    yaw_hist = deque(maxlen=20)
    blink_times = deque()
    blink_count = 0
    consec_ear = 0
    session_start = time.time()

    arm_hist = deque(maxlen=SMOOTH_WINDOW)
    shoulder_adv_hist = deque(maxlen=SMOOTH_WINDOW)
    neck_hist_buf = deque(maxlen=SMOOTH_WINDOW)
    sitting_hist = deque(maxlen=SMOOTH_WINDOW)
    cheek_hist = deque(maxlen=SMOOTH_WINDOW)
    lip_move_hist = deque(maxlen=10)
    ear_hist_buf = deque(maxlen=10)

    frame_n = 0
    last_report_time = time.time()
    latest_report = {}

    sig = {
        "gaze": "-", "eye_contact_score": 0.5,
        "head_pose": "-", "nodding": False, "head_shake": False,
        "blinks_per_minute": 0,
        "brow_score": 0.5, "brow_label": "-",
        "lip_score": 0.5, "lip_label": "-",
        "smile_genuine": False, "smile_label": "-",
        "micro_tension_score": 0, "engagement_score": 5,
        "face_detected": False,
        "body_language": {
            "arms": "RELAXED",
            "shoulders": {"alignment": "STRAIGHT", "energy": "ACTIVE", "position": "NEUTRAL"},
            "neck": {"position": "STRAIGHT", "stability": "STABLE"},
            "sitting_posture": "UPRIGHT",
        },
        "face": {
            "cheeks": "RELAXED",
            "lips": {"state": "RELAXED", "movement": "LOW"},
        },
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read camera.")
            break

        frame_n += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Face Mesh ──────────────────────────────────
        fm = fmesh.process(rgb)
        if fm.multi_face_landmarks:
            lms = fm.multi_face_landmarks[0].landmark
            sig["face_detected"] = True

            sig["gaze"] = get_gaze(lms, w, h)
            sig["eye_contact_score"] = get_eye_contact(lms)

            yaw, pitch, pose_label = get_head_pose(lms)
            sig["head_pose"] = pose_label
            pitch_hist.append(pitch)
            yaw_hist.append(yaw)
            sig["nodding"] = get_nod(pitch_hist)
            sig["head_shake"] = get_shake(yaw_hist)

            # Blink detection
            l_ear_val = ear(lms, LEFT_EYE_IDX, w, h)
            r_ear_val = ear(lms, RIGHT_EYE_IDX, w, h)
            avg_ear = (l_ear_val + r_ear_val) / 2.0
            if avg_ear < EAR_THRESHOLD:
                consec_ear += 1
            else:
                if consec_ear >= EAR_CONSEC_FRAMES:
                    blink_count += 1
                    blink_times.append(time.time())
                consec_ear = 0

            now = time.time()
            while blink_times and now - blink_times[0] > BLINK_WINDOW_SEC:
                blink_times.popleft()
            elapsed = min(now - session_start, BLINK_WINDOW_SEC)
            sig["blinks_per_minute"] = (len(blink_times) / elapsed * 60) if elapsed > 0 else 0

            bs, bl_label = get_brow(lms)
            sig["brow_score"], sig["brow_label"] = bs, bl_label

            sg, sl = get_smile(lms, ear_hist_buf)
            sig["smile_genuine"], sig["smile_label"] = sg, sl

            # Cheek
            raw_cheek = get_cheek_state(lms)
            cheek_hist.append(raw_cheek)
            smoothed_cheek = _mode_vote(cheek_hist)

            # Lip
            lip_result, lip_s = get_lip(lms, lip_move_hist)
            sig["lip_score"] = lip_s
            sig["lip_label"] = lip_result["state"]

            sig["face"] = {
                "cheeks": smoothed_cheek,
                "lips": lip_result,
            }

            sig["micro_tension_score"] = tension_score(
                bs, lip_s, sig["blinks_per_minute"], "neutral",
            )
            sig["engagement_score"] = engagement_score_calc(sig)

        else:
            sig["face_detected"] = False

        # ── Pose ────────────────────────────────────────
        pose_results = pose_model.process(rgb)
        if pose_results.pose_landmarks:
            plm = pose_results.pose_landmarks.landmark

            raw_arm = detect_arm_state(plm)
            arm_hist.append(raw_arm)
            smoothed_arm = _mode_vote(arm_hist)

            raw_shoulder = detect_shoulder_advanced(plm)
            shoulder_adv_hist.append(raw_shoulder["alignment"])
            smoothed_sh_align = _mode_vote(shoulder_adv_hist)
            smoothed_shoulder = {
                "alignment": smoothed_sh_align,
                "energy": raw_shoulder["energy"],
                "position": raw_shoulder["position"],
            }

            raw_neck = detect_neck_position(plm)
            neck_hist_buf.append(raw_neck["position"])
            smoothed_neck_pos = _mode_vote(neck_hist_buf)
            smoothed_neck = {
                "position": smoothed_neck_pos,
                "stability": raw_neck["stability"],
            }

            raw_sitting = detect_sitting_posture(plm)
            sitting_hist.append(raw_sitting)
            smoothed_sitting = _mode_vote(sitting_hist)

            sig["body_language"] = {
                "arms": smoothed_arm,
                "shoulders": smoothed_shoulder,
                "neck": smoothed_neck,
                "sitting_posture": smoothed_sitting,
            }

            # Draw pose skeleton
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            )

        # ── Hands ──────────────────────────────────────
        hands_results = hands_model.process(rgb)
        if hands_results.multi_hand_landmarks:
            for hlm in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # ── Event Tracking ──────────────────────────────
        event_tracker.update(sig)

        # ── Cumulative Report ───────────────────────────
        reporter.push(sig)

        now = time.time()
        if now - last_report_time >= REPORT_INTERVAL:
            latest_report = reporter.generate(sig, event_tracker)
            last_report_time = now

            # Post to backend
            backend_poster.push(latest_report)

        # ── Debug Overlay ───────────────────────────────
        active_events = event_tracker.get_active_events()
        draw_overlay(frame, sig, latest_report, active_events)
        cv2.imshow("Trusted Advisor AI - Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[Trusted Advisor AI] Quit requested.")
            break

    # Cleanup
    backend_poster.stop()
    cap.release()
    fmesh.close()
    pose_model.close()
    hands_model.close()
    cv2.destroyAllWindows()
    print("[Trusted Advisor AI] Done.")


if __name__ == "__main__":
    main()
