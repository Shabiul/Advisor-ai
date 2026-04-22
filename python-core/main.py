"""
=============================================================
  Trusted Advisor AI — Main Pipeline (V6)
  Full vision → analytics → HTTP → dashboard pipeline
  Based on the working reference camera_pipeline.py
=============================================================
"""

import os
import sys

# Fix Windows console encoding for Unicode symbols
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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
from http.server import HTTPServer, BaseHTTPRequestHandler
from session_recorder import SessionRecorder

# ─────────────────────────────────────────────
#  VIDEO POSTER (Node.js Integration — fallback)
# ─────────────────────────────────────────────
class VideoPoster:
    def __init__(self, url):
        self._url = url
        self._pending_frame = None
        self._lock = threading.Lock()
        self._running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def push(self, frame_bytes):
        with self._lock:
            self._pending_frame = frame_bytes

    def _run(self):
        while self._running:
            frame_to_send = None
            with self._lock:
                if self._pending_frame:
                    frame_to_send = self._pending_frame
                    self._pending_frame = None
            
            if frame_to_send:
                try:
                    req = urllib.request.Request(
                        self._url,
                        data=frame_to_send,
                        headers={"Content-Type": "image/jpeg"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=1)
                except Exception:
                    pass
            time.sleep(0.04)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────
#  DIRECT MJPEG STREAM SERVER (fast, no relay)
# ─────────────────────────────────────────────
MJPEG_PORT = int(os.environ.get("ADVISOR_MJPEG_PORT", "9090"))


class MJPEGServer:
    """Lightweight MJPEG HTTP server. Browsers connect directly to get
    the live stream at full FPS with zero relay overhead."""

    def __init__(self, port=MJPEG_PORT):
        self._port = port
        self._latest_jpeg = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._clients = []
        self._running = True

    def start(self):
        t = threading.Thread(target=self._serve, daemon=True)
        t.start()
        print(f"[MJPEG] Direct stream on http://localhost:{self._port}/video_feed")

    def push(self, jpeg_bytes):
        """Main loop pushes JPEG frames here."""
        with self._lock:
            self._latest_jpeg = jpeg_bytes
        self._event.set()

    def _serve(self):
        server = HTTPServer(("0.0.0.0", self._port), self._make_handler())
        server.timeout = 0.5
        while self._running:
            server.handle_request()

    def _make_handler(self):
        mjpeg_server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/video_feed":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    try:
                        while mjpeg_server._running:
                            mjpeg_server._event.wait(timeout=0.1)
                            mjpeg_server._event.clear()
                            with mjpeg_server._lock:
                                jpeg = mjpeg_server._latest_jpeg
                            if jpeg:
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b"MJPEG server. Use /video_feed")

            def handle(self):
                """Suppress ConnectionAbortedError on Windows when clients disconnect."""
                try:
                    super().handle()
                except (ConnectionAbortedError, ConnectionResetError, OSError):
                    pass

            def log_message(self, format, *args):
                pass  # suppress request logs

        return Handler

    def stop(self):
        self._running = False

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

# Audio emotion signal file (written by emo_service.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_SIGNAL_FILE = os.path.join(_PROJECT_ROOT, "ta_audio_signals.json")

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

    lip_h = abs(lms[14].y - lms[13].y) / face_h

    score = round(max(0.0, 1.0 - min(lip_h / 0.025, 1.0)), 2)

    if lip_h > 0.035:
        state = "SPEAKING"
    elif lip_h > 0.015:
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

    if mouth_w > 0.95 and eye_relaxed:
        return True, "GENUINE"
    if mouth_w > 0.95:
        return False, "SOCIAL"
    if mouth_w > 0.82:
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

        # ── Audio emotion fusion ──────────────────────
        audio_emo = sig.get("audio_emotion", {})
        multimodal = self._build_multimodal(sig, audio_emo, attention_score)

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
            # ── Audio + Multimodal fusion ──
            "audio_emotion": audio_emo if audio_emo else None,
            "multimodal_emotion": multimodal,
        }

        return report

    def _build_multimodal(self, sig, audio_emo, attention_score):
        """Fuse vision and audio signals into unified behavioral insights."""
        if not audio_emo or not audio_emo.get("label"):
            return None

        audio_label = audio_emo.get("label", "neutral")
        audio_conf = audio_emo.get("confidence", 0)
        audio_valence = audio_emo.get("valence", 0)
        audio_arousal = audio_emo.get("arousal", 0)

        # Vision emotional proxy from facial signals
        smile = sig.get("smile_genuine", False)
        smile_label = sig.get("smile_label", "NONE")
        brow_label = sig.get("brow_label", "NEUTRAL")
        lip_label = sig.get("lip_label", "RELAXED")
        tension = sig.get("micro_tension_score", 0)
        engagement = sig.get("engagement_score", 5)

        # Derive vision emotion from facial signals
        if smile and smile_label == "GENUINE":
            vision_emo = "happy"
        elif brow_label == "FURROWED" and tension >= 6:
            vision_emo = "angry"
        elif brow_label == "RAISED" and lip_label in ["SLIGHTLY_OPEN", "SPEAKING"]:
            vision_emo = "surprised"
        elif tension >= 5 and lip_label == "COMPRESSED":
            vision_emo = "fear"
        elif engagement <= 3 and not smile:
            vision_emo = "sad"
        else:
            vision_emo = "neutral"

        # Congruence: do face and voice agree?
        VALENCE_MAP = {
            "happy": 1.0, "calm": 0.7, "neutral": 0.5, "surprised": 0.4,
            "sad": -0.3, "fear": -0.5, "fearful": -0.5,
            "angry": -0.7, "disgust": -0.8,
        }
        v_vision = VALENCE_MAP.get(vision_emo, 0.0)
        v_audio = VALENCE_MAP.get(audio_label, 0.0)
        diff = abs(v_vision - v_audio)
        congruence = round(max(0.0, 1.0 - diff / 1.8), 2)

        # Fused emotion: weighted blend favoring higher-confidence source
        if vision_emo == audio_label:
            fused_label = audio_label
            fused_conf = min(1.0, audio_conf * 1.2)
        elif congruence > 0.6:
            fused_label = audio_label if audio_conf > 0.5 else vision_emo
            fused_conf = audio_conf
        else:
            fused_label = audio_label  # voice is harder to fake
            fused_conf = audio_conf * 0.8

        # Behavioral insights
        insights = []
        if congruence >= 0.8:
            insights.append(f"Strong emotional congruence — face and voice both indicate {fused_label}.")
        elif congruence >= 0.5:
            insights.append(f"Mixed signals — face reads {vision_emo} but voice indicates {audio_label}. Possible social masking.")
        else:
            insights.append(f"Emotional mismatch — face shows {vision_emo}, voice reveals {audio_label}. Client may be suppressing true emotions.")

        if audio_arousal > 0.3 and engagement <= 4:
            insights.append("High vocal arousal but low visual engagement — possible internal distress or frustration.")
        if audio_label in ["angry", "fear", "disgust"] and smile:
            insights.append("Negative vocal emotion with visible smile — potential stress response or nervous laughter.")
        if audio_valence < -0.2 and tension >= 5:
            insights.append("Both voice and facial muscles indicate stress — recommend a pause or topic change.")
        if audio_label == "happy" and engagement >= 7:
            insights.append("Positive vocal tone with high engagement — client is genuinely receptive and comfortable.")

        return {
            "vision_emotion": vision_emo,
            "audio_emotion": audio_label,
            "audio_confidence": round(audio_conf, 2),
            "fused_emotion": fused_label,
            "fused_confidence": round(fused_conf, 2),
            "congruence": congruence,
            "congruence_level": "HIGH" if congruence >= 0.7 else "MEDIUM" if congruence >= 0.4 else "LOW",
            "vad": {
                "valence": audio_valence,
                "arousal": audio_arousal,
                "dominance": audio_emo.get("dominance", 0),
            },
            "behavioral_insights": insights,
        }

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

    def push(self, report, sig):
        with self._lock:
            self._pending = (report, sig)

    def _run(self):
        while self._running:
            item = None
            with self._lock:
                if self._pending is not None:
                    item = self._pending
                    self._pending = None

            if item is not None:
                try:
                    report, sig = item
                    payload = json.dumps(
                        {"mode": self._mode, "data": report, "sig": sig},
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
#  AUDIO SIGNAL READER (reads emo_service output)
# ─────────────────────────────────────────────
class AudioSignalReader:
    """Periodically reads ta_audio_signals.json written by emo_service.py
    and injects audio emotion data into the shared sig dict."""

    def __init__(self, sig, filepath=AUDIO_SIGNAL_FILE, interval=0.5):
        self._sig = sig
        self._filepath = filepath
        self._interval = interval
        self._running = True
        self._last_ts = 0
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        print(f"[AUDIO] Reader active — watching {filepath}")

    def _run(self):
        while self._running:
            try:
                if os.path.exists(self._filepath):
                    mtime = os.path.getmtime(self._filepath)
                    if mtime > self._last_ts:
                        self._last_ts = mtime
                        with open(self._filepath, "r") as f:
                            data = json.load(f)
                        self._inject(data)
            except (json.JSONDecodeError, IOError, OSError):
                pass
            time.sleep(self._interval)

    def _inject(self, data):
        """Inject audio emotion data into the shared signal dict."""
        ts = data.get("timestamp", 0)
        # Only inject if data is recent (within 5 seconds)
        if ts and (time.time() - ts) > 5:
            self._sig["audio_emotion"] = {}
            return

        self._sig["audio_emotion"] = {
            "label": data.get("label", "unknown"),
            "confidence": data.get("confidence", 0),
            "all_scores": data.get("all_scores", {}),
            "valence": data.get("valence", 0),
            "arousal": data.get("arousal", 0),
            "dominance": data.get("dominance", 0),
            "vad_quadrant": data.get("vad_quadrant", ""),
            "rms": data.get("rms", 0),
        }

        # Also inject transcript if available
        if data.get("transcript"):
            self._sig["live_transcript"] = data["transcript"]

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
#  ML WORKER (background thread)
# ─────────────────────────────────────────────
class MLWorker:
    """
    Runs heavy ML inference (FaceMesh, Pose, Hands) in a background thread.
    The main loop feeds it raw frames; it updates shared signal state.
    This decouples video streaming FPS from ML processing speed.
    """

    def __init__(self, sig, event_tracker, reporter, backend_poster, session_recorder):
        self._sig = sig
        self._event_tracker = event_tracker
        self._reporter = reporter
        self._backend_poster = backend_poster
        self._session_recorder = session_recorder
        self._pending_frame = None           # latest frame for ML
        self._lock = threading.Lock()
        self._running = True
        self._last_report_time = time.time()

        # ML-only state (owned exclusively by worker thread)
        self._pitch_hist = deque(maxlen=20)
        self._yaw_hist = deque(maxlen=20)
        self._blink_times = deque()
        self._blink_count = 0
        self._consec_ear = 0
        self._session_start = time.time()
        self._total_gestures = 0
        self._hands_prev = False
        self._arm_hist = deque(maxlen=SMOOTH_WINDOW)
        self._sh_hist = deque(maxlen=SMOOTH_WINDOW)
        self._neck_hist = deque(maxlen=SMOOTH_WINDOW)
        self._sit_hist = deque(maxlen=SMOOTH_WINDOW)
        self._cheek_hist = deque(maxlen=SMOOTH_WINDOW)
        self._lip_hist = deque(maxlen=10)
        self._ear_hist = deque(maxlen=10)

        # Latest ML drawing artifacts (pose/hands landmarks for overlay)
        self._pose_landmarks = None
        self._hand_landmarks_list = []
        self._latest_report = {}
        self._active_events = []

        # Init MediaPipe models (owned by this thread)
        self._fmesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._pose = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def push_frame(self, frame):
        """Main loop calls this to give the worker the latest frame."""
        with self._lock:
            self._pending_frame = frame

    def get_draw_state(self):
        """Main loop calls this to get latest ML results for drawing."""
        with self._lock:
            return (
                self._pose_landmarks,
                self._hand_landmarks_list,
                self._latest_report,
                self._active_events,
            )

    def _run(self):
        while self._running:
            # Grab latest frame
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Face Mesh ──────────────────────────────
            fm = self._fmesh.process(rgb)
            if fm.multi_face_landmarks:
                lms = fm.multi_face_landmarks[0].landmark
                self._sig["face_detected"] = True
                self._sig["gaze"] = get_gaze(lms, w, h)
                self._sig["eye_contact_score"] = get_eye_contact(lms)

                yaw, pitch, pose_label = get_head_pose(lms)
                self._sig["head_pose"] = pose_label
                self._pitch_hist.append(pitch)
                self._yaw_hist.append(yaw)
                self._sig["nodding"] = get_nod(self._pitch_hist)
                self._sig["head_shake"] = get_shake(self._yaw_hist)

                l_ear_val = ear(lms, LEFT_EYE_IDX, w, h)
                r_ear_val = ear(lms, RIGHT_EYE_IDX, w, h)
                avg_ear = (l_ear_val + r_ear_val) / 2.0
                if avg_ear < EAR_THRESHOLD:
                    self._consec_ear += 1
                else:
                    if self._consec_ear >= EAR_CONSEC_FRAMES:
                        self._blink_count += 1
                        self._blink_times.append(time.time())
                    self._consec_ear = 0

                now = time.time()
                while self._blink_times and now - self._blink_times[0] > BLINK_WINDOW_SEC:
                    self._blink_times.popleft()
                elapsed = min(now - self._session_start, BLINK_WINDOW_SEC)
                self._sig["blinks_per_minute"] = (len(self._blink_times) / elapsed * 60) if elapsed > 0 else 0

                bs, bl_label = get_brow(lms)
                self._sig["brow_score"], self._sig["brow_label"] = bs, bl_label

                sg, sl = get_smile(lms, self._ear_hist)
                self._sig["smile_genuine"], self._sig["smile_label"] = sg, sl

                raw_cheek = get_cheek_state(lms)
                self._cheek_hist.append(raw_cheek)
                smoothed_cheek = _mode_vote(self._cheek_hist)

                lip_result, lip_s = get_lip(lms, self._lip_hist)
                self._sig["lip_score"] = lip_s
                self._sig["lip_label"] = lip_result["state"]
                self._sig["face"] = {"cheeks": smoothed_cheek, "lips": lip_result}

                self._sig["micro_tension_score"] = tension_score(
                    bs, lip_s, self._sig["blinks_per_minute"], "neutral",
                )
                self._sig["engagement_score"] = engagement_score_calc(self._sig)
            else:
                self._sig["face_detected"] = False

            # ── Pose ───────────────────────────────────
            pose_results = self._pose.process(rgb)
            pose_lm = None
            if pose_results.pose_landmarks:
                plm = pose_results.pose_landmarks.landmark
                pose_lm = pose_results.pose_landmarks

                raw_arm = detect_arm_state(plm)
                self._arm_hist.append(raw_arm)
                smoothed_arm = _mode_vote(self._arm_hist)

                raw_shoulder = detect_shoulder_advanced(plm)
                self._sh_hist.append(raw_shoulder["alignment"])
                smoothed_sh_align = _mode_vote(self._sh_hist)
                smoothed_shoulder = {
                    "alignment": smoothed_sh_align,
                    "energy": raw_shoulder["energy"],
                    "position": raw_shoulder["position"],
                }

                raw_neck = detect_neck_position(plm)
                self._neck_hist.append(raw_neck["position"])
                smoothed_neck_pos = _mode_vote(self._neck_hist)
                smoothed_neck = {
                    "position": smoothed_neck_pos,
                    "stability": raw_neck["stability"],
                }

                raw_sitting = detect_sitting_posture(plm)
                self._sit_hist.append(raw_sitting)
                smoothed_sitting = _mode_vote(self._sit_hist)

                self._sig["body_language"] = {
                    "arms": smoothed_arm,
                    "shoulders": smoothed_shoulder,
                    "neck": smoothed_neck,
                    "sitting_posture": smoothed_sitting,
                }

            # ── Hands ──────────────────────────────────
            hands_results = self._hands.process(rgb)
            hands_present = False
            hand_lm_list = []
            if hands_results.multi_hand_landmarks:
                hands_present = True
                hand_lm_list = list(hands_results.multi_hand_landmarks)

            if hands_present and not self._hands_prev:
                self._total_gestures += 1
            self._hands_prev = hands_present
            self._sig["gestures"] = self._total_gestures

            # ── Events + Report ────────────────────────
            self._event_tracker.update(self._sig)
            self._reporter.push(self._sig)

            now = time.time()
            report = self._latest_report
            if now - self._last_report_time >= REPORT_INTERVAL:
                report = self._reporter.generate(self._sig, self._event_tracker)
                self._last_report_time = now
                self._backend_poster.push(report, self._sig)
                self._session_recorder.write_signals(self._sig, report)

            active = self._event_tracker.get_active_events()

            # Store latest drawing state (thread-safe)
            with self._lock:
                self._pose_landmarks = pose_lm
                self._hand_landmarks_list = hand_lm_list
                self._latest_report = report
                self._active_events = active

    def stop(self):
        self._running = False
        self._fmesh.close()
        self._pose.close()
        self._hands.close()


# ─────────────────────────────────────────────
#  MAIN LOOP  (streams at full camera FPS)
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
        return

    print("[Trusted Advisor AI] Starting V6 pipeline...")
    print(f"  Camera     : {CAMERA_INDEX}")
    print(f"  Backend URL: {BACKEND_URL}")
    print(f"  Mode       : {ADVISOR_MODE}")
    print("  Press 'q' in the camera window to quit.\n")

    # Shared signal dict — written by MLWorker, read by main loop
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
        "audio_emotion": {},    # injected by AudioSignalReader
        "live_transcript": [],  # injected by AudioSignalReader
    }

    event_tracker = EventTracker()
    reporter = CumulativeReporter(window_sec=10.0)
    backend_poster = BackendPoster(BACKEND_URL, ADVISOR_MODE)
    video_poster = VideoPoster(BACKEND_URL.replace("/analyze", "/video_frame"))

    # Audio emotion reader (reads from emo_service.py output file)
    audio_reader = AudioSignalReader(sig)

    # Direct MJPEG server (browser connects here — no relay)
    mjpeg_server = MJPEGServer(port=MJPEG_PORT)
    mjpeg_server.start()

    session_recorder = SessionRecorder(fps=30)
    session_recorder.start(frame_width=1280, frame_height=720)

    # Start ML in background thread
    ml_worker = MLWorker(sig, event_tracker, reporter, backend_poster, session_recorder)
    print("[PIPELINE] ML worker started in background thread")
    print("[PIPELINE] Audio signal reader watching for emo_service data")
    print("[PIPELINE] Video streaming at full camera FPS (uncapped)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read camera.")
            break

        # Record raw frame
        session_recorder.write_frame(frame)

        # Feed frame to ML worker (non-blocking — drops old frames)
        ml_worker.push_frame(frame.copy())

        # Get latest ML results for drawing (non-blocking)
        pose_lm, hand_lm_list, latest_report, active_events = ml_worker.get_draw_state()

        # Draw skeleton + hands from latest ML results onto current frame
        if pose_lm is not None:
            mp_drawing.draw_landmarks(
                frame, pose_lm, mp_pose.POSE_CONNECTIONS,
            )
        if hand_lm_list:
            for hlm in hand_lm_list:
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # Draw overlay with latest signals
        draw_overlay(frame, sig, latest_report, active_events)

        # JPEG encode once, push to both direct stream and Node relay
        ret_enc, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret_enc:
            jpeg_bytes = jpeg.tobytes()
            mjpeg_server.push(jpeg_bytes)   # direct to browser (fast)
            video_poster.push(jpeg_bytes)    # relay to Node (fallback)

    # Cleanup
    ml_worker.stop()
    audio_reader.stop()
    mjpeg_server.stop()
    session_recorder.stop()
    backend_poster.stop()
    video_poster.stop()
    cap.release()
    print("[Trusted Advisor AI] Done.")


if __name__ == "__main__":
    main()
