"""
=============================================================
  Trusted Advisor AI — Camera Pipeline (V5/V6)
  Behavioral intelligence engine with upper-body detection,
  attention classification, DeepFace emotion, gaze/blink
  tracking, head pose, tension & engagement scoring.
  * UPDATED: Cumulative Away Timer + Widened Thresholds + Protobuf Fix
=============================================================
"""

import os
import sys

# ---> PROTOBUF FIX ADDED HERE <---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Suppress TF noise before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPFACE_HOME"] = os.path.join(os.path.expanduser("~"), ".deepface")

import cv2
import numpy as np
import mediapipe as mp
import json
import time
import threading
from collections import deque, Counter
from deepface import DeepFace

# Import V6 pose analysis functions
# Resolve vision directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_VISION_DIR = os.path.join(_BASE_DIR, "python-core", "vision")
if _VISION_DIR not in sys.path:
    sys.path.insert(0, _VISION_DIR)

# Import Session Recorder for auto-recording
_RECORDER_DIR = os.path.join(_BASE_DIR, "python-core")
if _RECORDER_DIR not in sys.path:
    sys.path.insert(0, _RECORDER_DIR)
from session_recorder import SessionRecorder

from pose_analyzer import (
    detect_arm_state,
    detect_shoulder_advanced,
    detect_neck_position,
    detect_sitting_posture,
    detect_attention_state,
    interpret_behavior,
)


# ─────────────────────────────────────────────
#  TEMPORAL SMOOTHING
# ─────────────────────────────────────────────
SMOOTH_WINDOW = 5  # frames to average over


def _mode_vote(history):
    """Return the most common value from the last SMOOTH_WINDOW entries."""
    if not history:
        return None
    recent = list(history)[-SMOOTH_WINDOW:]
    counts = Counter(recent)
    return counts.most_common(1)[0][0]

# ─────────────────────────────────────────────
#  PATHS & CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "ta_face_signals.json")
LOG_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "session_facial_log.json")

print(f"[PIPELINE] Writing signals to: {DATA_FILE}")

CAMERA_INDEX = 0
EMOTION_EVERY_N = 20          # run DeepFace every N frames
EAR_THRESHOLD = 0.22          # eye-aspect-ratio blink threshold
EAR_CONSEC_FRAMES = 2         # consecutive sub-threshold frames → blink
BLINK_WINDOW_SEC = 60         # rolling window for blinks/min
AWAY_THRESHOLD_SECONDS = 3.0
# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark indices (468-point FaceMesh + refined irises)
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

# V6: Cheek region landmarks
L_CHEEK_IDX = [50, 101, 118, 36]
R_CHEEK_IDX = [280, 330, 347, 266]
CHEEK_LOWER_L = 36
CHEEK_LOWER_R = 266


# ─────────────────────────────────────────────
#  DEEPFACE — BACKGROUND EMOTION DETECTOR
# ─────────────────────────────────────────────
class EmotionDetector:
    """Runs DeepFace emotion analysis in a background thread."""

    def __init__(self):
        self._result = {
            "emotion": "loading",
            "emotion_conf": 0,
            "emotion_all": {},
        }
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._ready = False
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def push(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def get(self):
        with self._lock:
            return dict(self._result)

    def is_ready(self):
        return self._ready

    def _run(self):
        print("[EMOTION] Warming up DeepFace (first run may take ~10s)...")
        try:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(
                blank, actions=["emotion"],
                enforce_detection=False, silent=True,
                detector_backend="opencv",
            )
            self._ready = True
            print("[EMOTION] DeepFace ready!")
        except Exception as e:
            print(f"[EMOTION] Warmup failed: {e}")
            self._ready = True  # continue anyway

        while self._running:
            frame = None
            with self._lock:
                if self._frame is not None:
                    frame = self._frame
                    self._frame = None

            if frame is not None:
                try:
                    small = cv2.resize(frame, (320, 240))
                    result = DeepFace.analyze(
                        small, actions=["emotion"],
                        enforce_detection=False, silent=True,
                        detector_backend="opencv",
                    )
                    if isinstance(result, list):
                        result = result[0]

                    dom = result["dominant_emotion"]
                    emos = result["emotion"]
                    conf = emos.get(dom, 0)

                    with self._lock:
                        self._result = {
                            "emotion": dom,
                            "emotion_conf": round(conf, 1),
                            "emotion_all": {k: round(v, 1) for k, v in emos.items()},
                        }
                except Exception:
                    pass  # keep last result

            time.sleep(0.08)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────
#  SIGNAL FUNCTIONS
# ─────────────────────────────────────────────

def ear(landmarks, indices, w, h):
    """Eye Aspect Ratio (EAR) for blink detection."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C else 0.0


def get_gaze(lms, w, h):
    l_inner, l_outer = lms[133].x, lms[33].x
    l_iris  = lms[LEFT_IRIS_IDX].x
    l_eye_w = abs(l_outer - l_inner)
    l_ratio = (l_iris - min(l_inner, l_outer)) / l_eye_w if l_eye_w > 0.001 else 0.5

    r_inner, r_outer = lms[362].x, lms[263].x
    r_iris  = lms[RIGHT_IRIS_IDX].x
    r_eye_w = abs(r_outer - r_inner)
    r_ratio = (r_iris - min(r_inner, r_outer)) / r_eye_w if r_eye_w > 0.001 else 0.5

    avg_ratio = (l_ratio + r_ratio) / 2.0

    # BALANCED THRESHOLDS FOR GAZE
    if avg_ratio < 0.42:   # Moved up from 0.35
        return "LOOKING LEFT"
    elif avg_ratio > 0.58: # Moved down from 0.65
        return "LOOKING RIGHT"
    return "CENTER"


def get_eye_contact(lms):
    """Eye-contact score 0–1 (1 = direct, 0 = fully averted)."""
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
    nose      = lms[NOSE_TIP_IDX]
    eye_mid_x = (lms[L_EYE_C_IDX].x + lms[R_EYE_C_IDX].x) / 2
    eye_mid_y = (lms[L_EYE_C_IDX].y + lms[R_EYE_C_IDX].y) / 2
    yaw   = nose.x - eye_mid_x
    pitch = nose.y - eye_mid_y
    
    # BALANCED THRESHOLDS FOR HEAD POSE
    if abs(yaw) > 0.06:  # Was 0.08
        return yaw, pitch, "TURNED LEFT" if yaw < 0 else "TURNED RIGHT"
    if pitch > 0.30:     # Was 0.35 (Looking down)
        return yaw, pitch, "LOOKING DOWN"
    if pitch < 0.10:     # Was 0.05 (Looking up/Camera low)
        return yaw, pitch, "HEAD RAISED"
        
    return yaw, pitch, "FACING FORWARD"


def get_brow(lms):
    """Eyebrow raise score 0–1 + label (RAISED/FURROWED/NEUTRAL)."""
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
    # Wider range (0.02–0.12) for better sensitivity
    score = round(max(0.0, min((((l_dist + r_dist) / 2) - 0.02) / 0.10, 1.0)), 2)
    label = "RAISED" if score > 0.55 else "FURROWED" if score < 0.40 else "NEUTRAL"
    return score, label


def get_lip(lms, lip_hist=None):
    """V6: Rich lip analysis → dict {state, movement}."""
    face_h = abs(lms[FOREHEAD_IDX].y - lms[CHIN_IDX].y)
    if face_h < 0.001:
        return {"state": "RELAXED", "movement": "LOW"}, 0.5

    lip_h = abs(
        np.mean([lms[i].y for i in LOWER_LIP_IDX])
        - np.mean([lms[i].y for i in UPPER_LIP_IDX])
    ) / face_h

    score = round(max(0.0, 1.0 - min(lip_h / 0.04, 1.0)), 2)

    # Classify state
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

    # Track movement via lip_hist deque
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

    # Override state to SPEAKING if high movement + open lips
    if movement == "HIGH" and lip_h > 0.025:
        state = "SPEAKING"

    return {"state": state, "movement": movement}, score


def get_cheek_state(lms):
    """V6: Cheek analysis → RAISED / TENSE / RELAXED."""
    face_h = abs(lms[FOREHEAD_IDX].y - lms[CHIN_IDX].y)
    if face_h < 0.001:
        return "RELAXED"

    # Cheek lift: measure Y position of cheek landmarks relative to nose/eye
    l_cheek_y = np.mean([lms[i].y for i in L_CHEEK_IDX])
    r_cheek_y = np.mean([lms[i].y for i in R_CHEEK_IDX])
    cheek_mid_y = (l_cheek_y + r_cheek_y) / 2.0

    # Eye squeeze (low EAR = squinting, supports genuine smile)
    avg_ear_val = (ear(lms, LEFT_EYE_IDX, 1, 1) + ear(lms, RIGHT_EYE_IDX, 1, 1)) / 2

    # Mouth corner position
    mouth_y = (lms[LIP_LEFT_IDX].y + lms[LIP_RIGHT_IDX].y) / 2.0

    # Cheek-to-mouth gap (normalized)
    cheek_mouth_gap = (mouth_y - cheek_mid_y) / face_h

    # RAISED: cheeks pushed up (small gap + eye squeeze)
    if cheek_mouth_gap > 0.12 and avg_ear_val < 0.28:
        return "RAISED"

    # TENSE: compressed cheeks (small gap, no eye squeeze)
    if cheek_mouth_gap < 0.08:
        return "TENSE"

    return "RELAXED"


def get_smile(lms, ear_hist=None):
    face_w = abs(lms[L_EYE_C_IDX].x - lms[R_EYE_C_IDX].x)
    if face_w < 0.001:
        return False, "NONE"

    mouth_w = abs(lms[LIP_LEFT_IDX].x - lms[LIP_RIGHT_IDX].x) / face_w
    avg_ear = (ear(lms, LEFT_EYE_IDX, 1, 1) + ear(lms, RIGHT_EYE_IDX, 1, 1)) / 2

    if ear_hist is not None:
        ear_hist.append(avg_ear)

    eye_relaxed = False
    if ear_hist is not None and len(ear_hist) >= 3:
        recent = list(ear_hist)[-5:]
        ear_variance = max(recent) - min(recent)
        ear_mean = sum(recent) / len(recent)
        # Relaxed the eye-squint threshold (was 0.30, now 0.33)
        eye_relaxed = ear_variance < 0.04 and ear_mean < 0.33

    # TIGHTENED SMILE RATIOS (was 1.20, now 1.10)
    if mouth_w > 1.10 and eye_relaxed:
        return True, "GENUINE"
    if mouth_w > 1.10:
        return False, "SOCIAL"
    if mouth_w > 1.02: # Was 1.05
        return False, "SUBTLE"

    return False, "NONE"


def get_nod(pitch_hist):
    """Detect nodding from recent pitch oscillation."""
    if len(pitch_hist) < 6:
        return False
    recent = list(pitch_hist)[-6:]
    return np.mean([abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]) > 0.008


def get_shake(yaw_hist):
    """Detect head shake from recent yaw direction changes."""
    if len(yaw_hist) < 8:
        return False
    recent = list(yaw_hist)[-8:]
    changes = sum(
        1 for i in range(2, len(recent))
        if (recent[i] - recent[i - 1]) * (recent[i - 1] - recent[i - 2]) < 0
    )
    return changes >= 3


# ─────────────────────────────────────────────
#  COMPOSITE SCORES
# ─────────────────────────────────────────────

def tension_score(brow_s, lip_s, bpm, emotion):
    """Micro-tension score 0–10."""
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


def engagement_score(sig):
    """Rich engagement score 0–10 factoring emotion, gaze, blinks, tension."""
    emo = sig.get("emotion", "neutral").lower()
    gaze = sig.get("gaze", "CENTER")
    bpm = sig.get("blinks_per_minute", 15)
    ec = sig.get("eye_contact_score", 0.5)
    ten = sig.get("micro_tension_score", 0)

    # If emotion not yet loaded, score on physical signals only
    if emo in ["loading", "analyzing"]:
        s = 5
        if gaze == "CENTER":
            s += 1
        if ec > 0.7:
            s += 1
        s -= ten // 3
        return max(0, min(s, 10))

    s = 5
    if "happy" in emo:
        s += 2
    if "surprise" in emo:
        s += 1
    if "neutral" in emo:
        s += 0
    if "angry" in emo:
        s -= 3
    if "fear" in emo:
        s -= 2
    if "sad" in emo:
        s -= 2
    if "disgust" in emo:
        s -= 2
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
#  POSTURE (via MediaPipe Pose)
# ─────────────────────────────────────────────

def extract_posture(pose_landmarks):
    """Return 'Open' or 'Closed' based on shoulder vs hip width."""
    try:
        ls = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        rh = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        shoulder_w = abs(ls.x - rs.x)
        hip_w = abs(lh.x - rh.x)
        return "Open" if shoulder_w >= hip_w * 0.9 else "Closed"
    except (IndexError, AttributeError):
        return "Closed"


# ─────────────────────────────────────────────
#  HAND GESTURES (via MediaPipe Hands)
# ─────────────────────────────────────────────

def count_raised_fingers(hand_landmarks):
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]
    count = 0
    try:
        for tip, pip_ in zip(tips, pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip_].y:
                count += 1
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        if thumb_tip.x < thumb_ip.x:
            count += 1
    except (IndexError, AttributeError):
        pass
    return count


def extract_gestures(hands_results):
    total = 0
    if hands_results.multi_hand_landmarks:
        for hand_lm in hands_results.multi_hand_landmarks:
            total += count_raised_fingers(hand_lm)
    return int(total)


# ─────────────────────────────────────────────
#  NUMPY-SAFE JSON ENCODER
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  JSON WRITER (Windows-safe)
# ─────────────────────────────────────────────

def write_signals(sig):
    """Write signals dict to live_data.json (Windows-safe fallback)."""
    sig["timestamp"] = int(time.time())
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

    try:
        # Try atomic swap first
        tmp_path = DATA_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(sig, f, indent=2, cls=NumpyEncoder)
        os.replace(tmp_path, DATA_FILE)
    except OSError:
        # Fallback: direct write (Windows lock conflicts)
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(sig, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            print(f"[ERROR] Write failed: {e}")


# ─────────────────────────────────────────────
#  DEBUG OVERLAY
# ─────────────────────────────────────────────

def draw_debug(frame, sig, ready):
    h, w = frame.shape[:2]
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

    put("TRUSTED ADVISOR AI  V6", (80, 200, 255), bold=True)
    put("-" * 35, (40, 60, 80))

    if not ready:
        put("DeepFace loading...", (200, 165, 0))
    else:
        emo = sig.get("emotion", "-").upper()
        conf = sig.get("emotion_conf", 0)
        ecol = {
            "HAPPY": (0, 220, 100), "ANGRY": (0, 50, 255),
            "FEAR": (128, 0, 200), "SAD": (200, 120, 0),
            "SURPRISE": (0, 200, 255), "NEUTRAL": (180, 180, 180),
        }.get(emo, (200, 200, 200))
        put(f"EMOTION: {emo}", ecol, bold=True)
        put(f"CONF:    {conf:.0f}%", (150, 150, 150))

    put("-" * 35, (40, 60, 80))
    put(f"GAZE:    {sig.get('gaze', '-')}", (0, 200, 180))
    put(f"POSE:    {sig.get('head_pose', '-')}", (0, 180, 220))
    put(f"BLINKS:  {sig.get('blinks_per_minute', 0):.0f}/min", (180, 180, 180))
    
    # ---> UPDATED AWAY TIMERS DISPLAY <---
    put(f"CURR AWAY: {sig.get('current_looking_away_seconds', 0):.1f}s", (200, 165, 0))
    put(f"TOT AWAY:  {sig.get('total_away_time_seconds', 0):.1f}s", (255, 100, 50), bold=True)
    put(f"AWAYS #:   {sig.get('total_away_events', 0)}", (255, 150, 50))
    put("-" * 35, (40, 60, 80))

    # V6: body language
    bl = sig.get("body_language", {})
    put(f"ARMS:    {bl.get('arms', '-')}", (200, 200, 100))
    sh = bl.get("shoulders", {})
    if isinstance(sh, dict):
        put(f"SH-ALGN: {sh.get('alignment', '-')}", (200, 200, 100))
        put(f"SH-ENRG: {sh.get('energy', '-')}", (200, 200, 100))
        put(f"SH-POS:  {sh.get('position', '-')}", (200, 200, 100))
    nk = bl.get("neck", {})
    if isinstance(nk, dict):
        nk_col = (0, 220, 100) if nk.get('position') == 'STRAIGHT' else (200, 165, 0)
        put(f"NECK:    {nk.get('position', '-')} [{nk.get('stability', '-')}]", nk_col)
    sit = bl.get("sitting_posture", "-")
    sit_col = (0, 220, 100) if "UPRIGHT" in sit else (0, 180, 220) if "ENGAGED" in sit else (200, 165, 0)
    put(f"SIT:     {sit}", sit_col)

    attn = sig.get("attention_state", "-")
    attn_conf = sig.get("attention_confidence", 0)
    attn_col = (
        (0, 220, 100) if attn == "ATTENTIVE"
        else (200, 165, 0) if attn == "NEUTRAL"
        else (0, 50, 255)
    )
    put(f"ATTN:    {attn} ({attn_conf:.0%})", attn_col, bold=True)
    put("-" * 35, (40, 60, 80))

    # V6: face details
    face = sig.get("face", {})
    put(f"CHEEKS:  {face.get('cheeks', '-')}", (180, 180, 180))
    lips = face.get("lips", {})
    if isinstance(lips, dict):
        put(f"LIPS:    {lips.get('state', '-')} [{lips.get('movement', '-')}]", (180, 180, 180))

    brow_l = sig.get("brow_label", "-")
    smile_l = sig.get("smile_label", "-")
    put(f"BROW:    {brow_l} ({sig.get('brow_score', 0):.2f})", (180, 180, 180))
    put(f"SMILE:   {smile_l}", (180, 180, 180))
    nod_str = "YES" if sig.get("nodding") else "NO"
    shake_str = "YES" if sig.get("head_shake") else "NO"
    put(f"NOD: {nod_str}  SHAKE: {shake_str}", (180, 180, 180))
    put("-" * 35, (40, 60, 80))

    eng = sig.get("engagement_score", 0)
    ten = sig.get("micro_tension_score", 0)
    put(f"ENGAGE:  {eng}/10", (0, 220, 100) if eng >= 6 else (200, 165, 0))
    put(f"TENSION: {ten}/10", (0, 50, 255) if ten >= 6 else (0, 220, 100))

    # V6: behavior tags
    tags = sig.get("behavior_tags", [])
    tag_str = " | ".join(tags) if tags else "-"
    put(f"BEHAV:   {tag_str}", (80, 200, 255), bold=True)

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
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. Check your webcam connection.")
        return

    print("[PIPELINE] Starting... press Q to quit.")

    emo_det = EmotionDetector()
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

    pitch_hist = deque(maxlen=20)
    yaw_hist = deque(maxlen=20)
    blink_times = deque()
    blink_count = 0
    consec_ear = 0
    frame_n = 0
    session_start = time.time()
    log = []

    # ── Session Recorder (auto-start) ──────────────────
    session_recorder = SessionRecorder(
        base_dir=os.path.join(BASE_DIR, "recordings"),
        fps=30,
    )
    session_recorder.start(frame_width=1280, frame_height=720)

    # ---> NEW: Tracking variables for Cumulative Timer <---
    away_start = None
    max_away = 0
    total_away_time = 0.0  
    away_events = []       
    focus_counter = 0
    away_counter = 0
    state = "FOCUSED"

    # V6: temporal smoothing buffers
    arm_hist = deque(maxlen=SMOOTH_WINDOW)
    shoulder_adv_hist = deque(maxlen=SMOOTH_WINDOW)
    neck_hist = deque(maxlen=SMOOTH_WINDOW)
    sitting_hist = deque(maxlen=SMOOTH_WINDOW)
    attention_hist = deque(maxlen=SMOOTH_WINDOW)
    confidence_hist = deque(maxlen=SMOOTH_WINDOW)
    cheek_hist = deque(maxlen=SMOOTH_WINDOW)
    lip_move_hist = deque(maxlen=10)
    behavior_hist = deque(maxlen=SMOOTH_WINDOW)
    ear_hist = deque(maxlen=10)       # for eye relaxation in smile detection

    sig = {
        "emotion": "loading", "emotion_conf": 0, "emotion_all": {},
        "gaze": "-", "eye_contact_score": 0.5,
        "head_pose": "-", "nodding": False, "head_shake": False,
        "blinks_per_minute": 0,
        "brow_score": 0.5, "brow_label": "-",
        "lip_score": 0.5, "lip_label": "-",
        "smile_genuine": False, "smile_label": "-",
        "micro_tension_score": 0, "engagement_score": 5,
        "posture": "Closed", "gestures": 0,
        # V6: body language & face sub-dicts
        "body_language": {
            "arms": "RELAXED",
            "shoulders": {"alignment": "STRAIGHT", "energy": "ACTIVE", "position": "NEUTRAL"},
            "neck": {"position": "STRAIGHT", "stability": "STABLE"},
            "sitting_posture": "UPRIGHT_CONFIDENT",
            "posture": "Closed",
        },
        "face": {
            "cheeks": "RELAXED",
            "lips": {"state": "RELAXED", "movement": "LOW"},
        },
        "attention_state": "NEUTRAL",
        "attention_confidence": 0.5,
        "behavior_tags": ["OBSERVING"],
        
        # New default values for the UI
        "current_looking_away_seconds": 0,
        "max_away_seconds": 0,
        "total_away_time_seconds": 0,
        "total_away_events": 0,
        "away_history": [],
        "proctor_alert": False
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] Cannot read camera {CAMERA_INDEX}.")
            break

        # Record raw frame before any overlay/drawing
        session_recorder.write_frame(frame)

        frame_n += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── DeepFace (async) ──────────────────────────
        if emo_det.is_ready() and frame_n % EMOTION_EVERY_N == 0:
            emo_det.push(frame)
        emo_data = emo_det.get()
        sig["emotion"] = emo_data["emotion"]
        sig["emotion_conf"] = emo_data["emotion_conf"]
        sig["emotion_all"] = emo_data["emotion_all"]

        # ── Face Mesh ─────────────────────────────────
        fm = fmesh.process(rgb)
        if fm.multi_face_landmarks:
            lms = fm.multi_face_landmarks[0].landmark

            sig["gaze"] = get_gaze(lms, w, h)
            sig["eye_contact_score"] = get_eye_contact(lms)

            yaw, pitch, pose_label = get_head_pose(lms)
            sig["head_pose"] = pose_label
            pitch_hist.append(pitch)
            yaw_hist.append(yaw)
            sig["nodding"] = get_nod(pitch_hist)
            sig["head_shake"] = get_shake(yaw_hist)
            
            # ---> NEW: Cumulative Timer Logic <---
            is_focused = (sig["gaze"] == "CENTER" and sig["head_pose"] == "FACING FORWARD")

            if is_focused:
                focus_counter += 1
                away_counter = 0
            else:
                away_counter += 1
                focus_counter = 0

            if focus_counter >= 2:
                state = "FOCUSED"
            elif away_counter >= 3:
                state = "AWAY"

            now = time.time()
            live_total_away = total_away_time

            if state == "AWAY":
                if away_start is None:
                    away_start = now
                current_away = now - away_start
                live_total_away = total_away_time + current_away
            else:
                if away_start is not None:
                    duration = now - away_start
                    max_away = max(max_away, duration)
                    
                    if duration > 0.5: 
                        total_away_time += duration  
                        away_events.append({
                            "timestamp": round(now - session_start, 1),
                            "duration_seconds": round(duration, 2)
                        })
                    
                    away_start = None
                
                current_away = 0
                live_total_away = total_away_time 
            
            sig["current_looking_away_seconds"] = round(current_away, 2)
            sig["max_away_seconds"] = round(max_away, 2)
            sig["total_away_time_seconds"] = round(live_total_away, 2)
            sig["total_away_events"] = len(away_events)
            sig["away_history"] = away_events[-5:] 
            sig["proctor_alert"] = (current_away >= AWAY_THRESHOLD_SECONDS)
            
            # Debug every 30 frames
            if frame_n % 30 == 0:
                print(
                    f"[DEBUG] gaze={sig['gaze']}  ec={sig['eye_contact_score']}  "
                    f"eng={sig.get('engagement_score', 0)}  ten={sig.get('micro_tension_score', 0)}  "
                    f"emo={sig.get('emotion', '?')}  tot_away={sig['total_away_time_seconds']}s"
                )

            # Blink detection via EAR
            l_ear = ear(lms, LEFT_EYE_IDX, w, h)
            r_ear = ear(lms, RIGHT_EYE_IDX, w, h)
            avg_ear = (l_ear + r_ear) / 2.0
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

            bs, bl = get_brow(lms)
            sig["brow_score"], sig["brow_label"] = bs, bl

            # V6: lip detection moved below (cheek section) — use current lip_score for tension
            ls = sig.get("lip_score", 0.5)

            sg, sl = get_smile(lms, ear_hist)
            sig["smile_genuine"], sig["smile_label"] = sg, sl

            sig["micro_tension_score"] = tension_score(
                bs, ls, sig["blinks_per_minute"], sig["emotion"],
            )

            # V6: cheek state
            raw_cheek = get_cheek_state(lms)
            cheek_hist.append(raw_cheek)
            smoothed_cheek = _mode_vote(cheek_hist)

            # V6: advanced lip
            lip_result, lip_s = get_lip(lms, lip_move_hist)
            sig["lip_score"] = lip_s
            sig["lip_label"] = lip_result["state"]

            sig["face"] = {
                "cheeks": smoothed_cheek,
                "lips": lip_result,
            }

            sig["engagement_score"] = engagement_score(sig)

        # ── Pose ──────────────────────────────────────
        pose_results = pose_model.process(rgb)
        if pose_results.pose_landmarks:
            plm = pose_results.pose_landmarks.landmark
            sig["posture"] = extract_posture(pose_results)

            # V6: arm state (with smoothing)
            raw_arm = detect_arm_state(plm)
            arm_hist.append(raw_arm)
            smoothed_arm = _mode_vote(arm_hist)

            # V6: advanced shoulder analysis
            raw_shoulder = detect_shoulder_advanced(plm)
            shoulder_adv_hist.append(raw_shoulder["alignment"])
            smoothed_sh_align = _mode_vote(shoulder_adv_hist)
            smoothed_shoulder = {
                "alignment": smoothed_sh_align,
                "energy": raw_shoulder["energy"],
                "position": raw_shoulder["position"],
            }

            # V6: neck position
            raw_neck = detect_neck_position(plm)
            neck_hist.append(raw_neck["position"])
            smoothed_neck_pos = _mode_vote(neck_hist)
            smoothed_neck = {
                "position": smoothed_neck_pos,
                "stability": raw_neck["stability"],
            }

            # V6: sitting posture
            raw_sitting = detect_sitting_posture(plm)
            sitting_hist.append(raw_sitting)
            smoothed_sitting = _mode_vote(sitting_hist)

            # V6: attention state
            face_data = {
                "eye_contact_score": sig.get("eye_contact_score", 0),
                "head_pose": sig.get("head_pose", ""),
            }
            pose_data = {
                "shoulders": smoothed_shoulder,
                "arms": smoothed_arm,
            }
            raw_attn, raw_conf = detect_attention_state(face_data, pose_data)
            attention_hist.append(raw_attn)
            confidence_hist.append(raw_conf)
            smoothed_attn = _mode_vote(attention_hist)
            smoothed_conf = round(
                sum(confidence_hist) / len(confidence_hist), 2
            ) if confidence_hist else 0.5

            # Update signal dict with V6 body language
            sig["body_language"] = {
                "arms": smoothed_arm,
                "shoulders": smoothed_shoulder,
                "neck": smoothed_neck,
                "sitting_posture": smoothed_sitting,
                "posture": sig["posture"],
            }
            sig["attention_state"] = smoothed_attn
            sig["attention_confidence"] = smoothed_conf

            # V6: behavioral interpretation
            raw_tags = interpret_behavior(sig)
            behavior_hist.append(tuple(raw_tags))
            # Use most recent tags (list isn't mode-votable)
            sig["behavior_tags"] = raw_tags

            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            )

        # ── Hands ─────────────────────────────────────
        hands_results = hands_model.process(rgb)
        sig["gestures"] = extract_gestures(hands_results)
        if hands_results.multi_hand_landmarks:
            for hlm in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # ── Write JSON ────────────────────────────────
        write_signals(sig)

        # ── Record signals to session log ──────────────
        session_recorder.write_signals(sig)

        # ── Debug overlay ─────────────────────────────
        draw_debug(frame, sig, emo_det.is_ready())
        cv2.imshow("Trusted Advisor AI - Press Q to quit", frame)

        # ── Session log every ~5s (at 30fps) ──────────
        if frame_n % 150 == 0:
            log.append({"ts": round(time.time() - session_start, 1), **sig})

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[SAVED] {fname}")

    # Cleanup
    session_recorder.stop()
    emo_det.stop()
    cap.release()
    fmesh.close()
    pose_model.close()
    hands_model.close()
    cv2.destroyAllWindows()

    os.makedirs(os.path.dirname(LOG_OUTPUT_PATH), exist_ok=True)
    with open(LOG_OUTPUT_PATH, "w") as f:
        json.dump(log, f, indent=2, cls=NumpyEncoder)
    print(f"[DONE] Log → {LOG_OUTPUT_PATH} | Total blinks: {blink_count}")


if __name__ == "__main__":
    main()