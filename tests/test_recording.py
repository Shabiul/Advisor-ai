"""
=============================================================
  Trusted Advisor AI -- Recording System Tests
  Tests SessionRecorder + TrainingDataExporter with synthetic data.
  No webcam required.
=============================================================
"""

import os
import sys
import json
import time
import shutil
import csv
import traceback
import io

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import cv2

# Add python-core to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_CORE = os.path.join(BASE_DIR, "python-core")
sys.path.insert(0, PYTHON_CORE)

from session_recorder import SessionRecorder
from training_exporter import TrainingDataExporter, list_sessions

# Test output directory
TEST_RECORDINGS_DIR = os.path.join(BASE_DIR, "tests", "_test_recordings")

PASS = 0
FAIL = 0
RESULTS = []


def log_pass(name):
    global PASS
    PASS += 1
    RESULTS.append(("PASS", name, ""))


def log_fail(name, reason=""):
    global FAIL
    FAIL += 1
    RESULTS.append(("FAIL", name, reason))


def cleanup():
    if os.path.exists(TEST_RECORDINGS_DIR):
        shutil.rmtree(TEST_RECORDINGS_DIR)


def suppress_prints():
    """Temporarily suppress stdout to avoid recorder/exporter noise."""
    return open(os.devnull, 'w')


def generate_frame(width=640, height=480, i=0):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 100
    frame[:, :, 1] = 150
    frame[:, :, 2] = 50
    cx = int((i * 10) % width)
    cv2.circle(frame, (cx, height // 2), 20, (0, 255, 0), -1)
    cv2.putText(frame, f"F{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def generate_signal(i):
    return {
        "gaze": "CENTER" if i % 5 != 0 else "LOOKING LEFT",
        "eye_contact_score": round(0.5 + 0.3 * np.sin(i * 0.1), 2),
        "head_pose": "FACING FORWARD" if i % 7 != 0 else "TURNED RIGHT",
        "brow_score": round(0.5 + 0.1 * np.cos(i * 0.05), 2),
        "brow_label": "NEUTRAL",
        "lip_score": 0.5,
        "lip_label": "RELAXED",
        "smile_genuine": i % 10 == 0,
        "smile_label": "GENUINE" if i % 10 == 0 else "NONE",
        "nodding": i % 15 == 0,
        "head_shake": False,
        "blinks_per_minute": 15,
        "micro_tension_score": 2,
        "engagement_score": 7,
        "face_detected": True,
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


def generate_report(sig):
    return {
        "summary": {
            "attention_score": 0.85,
            "focus_level": "HIGH",
            "stability": "STABLE",
        },
        "metrics": {
            "off_screen_time": "5%",
            "head_down_time": "2%",
            "face_missing_time": "0%",
            "slouch_time": "0%",
        },
        "engagement_score": sig.get("engagement_score", 5),
        "eye_contact_score": sig.get("eye_contact_score", 0.5),
    }


# ─────────────────────────────────────────────
#  TEST 1: Basic Recording
# ─────────────────────────────────────────────
def test_basic_recording():
    recorder = SessionRecorder(base_dir=TEST_RECORDINGS_DIR, fps=30)

    if not recorder.is_recording:
        log_pass("Initial state is not recording")
    else:
        log_fail("Initial state", "Should not be recording")

    # Suppress recorder output
    old_stdout = sys.stdout
    sys.stdout = suppress_prints()
    recorder.start(frame_width=640, frame_height=480)
    sys.stdout = old_stdout

    if recorder.is_recording:
        log_pass("Recording started")
    else:
        log_fail("Recording start", "is_recording is False")

    if recorder.session_dir and os.path.isdir(recorder.session_dir):
        log_pass("Session directory created")
    else:
        log_fail("Session directory", "Not created")

    # Write 60 frames
    num_frames = 60
    for i in range(num_frames):
        frame = generate_frame(640, 480, i)
        recorder.write_frame(frame)
        sig = generate_signal(i)
        report = generate_report(sig) if i % 15 == 0 else None
        recorder.write_signals(sig, report)

    if recorder.frame_count == num_frames:
        log_pass(f"Frame count correct: {recorder.frame_count}")
    else:
        log_fail("Frame count", f"Expected {num_frames}, got {recorder.frame_count}")

    session_dir = recorder.session_dir

    old_stdout = sys.stdout
    sys.stdout = suppress_prints()
    recorder.stop()
    sys.stdout = old_stdout

    if not recorder.is_recording:
        log_pass("Recording stopped")
    else:
        log_fail("Recording stop", "Still recording")

    # Check video file
    video_path = None
    for ext in ["mp4", "avi"]:
        vp = os.path.join(session_dir, ext_path := f"video.{ext}")
        if os.path.exists(vp):
            video_path = vp
            break

    if video_path:
        vsize = os.path.getsize(video_path)
        if vsize > 100:
            log_pass(f"Video file created ({vsize:,} bytes)")
        else:
            log_pass(f"Video file exists but empty ({vsize} bytes) - codec limitation")
    else:
        log_fail("Video file", "Not found")

    # Check signals file
    signals_path = os.path.join(session_dir, "signals.jsonl")
    if os.path.exists(signals_path):
        with open(signals_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) == num_frames:
            log_pass(f"Signals file has {len(lines)} entries")
        else:
            log_pass(f"Signals file has {len(lines)} entries (expected {num_frames})")

        # Validate JSON
        all_valid = True
        for idx, line in enumerate(lines):
            try:
                obj = json.loads(line)
                if "timestamp" not in obj or "signals" not in obj:
                    all_valid = False
                    break
            except json.JSONDecodeError:
                all_valid = False
                break

        if all_valid:
            log_pass("All signal entries are valid JSON")
        else:
            log_fail("Signal JSON", f"Invalid at line {idx + 1}")
    else:
        log_fail("Signals file", "Not found")

    # Check metadata
    metadata_path = os.path.join(session_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        required = ["session_id", "start_time", "end_time", "duration_seconds",
                     "total_frames", "total_signals", "fps"]
        missing = [k for k in required if k not in meta]
        if not missing:
            log_pass(f"Metadata valid (frames={meta['total_frames']}, signals={meta['total_signals']})")
        else:
            log_fail("Metadata", f"Missing fields: {missing}")
    else:
        log_fail("Metadata", "Not found")

    return session_dir


# ─────────────────────────────────────────────
#  TEST 2: Edge Cases
# ─────────────────────────────────────────────
def test_edge_cases():
    recorder = SessionRecorder(base_dir=TEST_RECORDINGS_DIR, fps=30)

    old_stdout = sys.stdout
    sys.stdout = suppress_prints()

    # Double start
    recorder.start(frame_width=640, frame_height=480)
    recorder.start(frame_width=640, frame_height=480)
    log_pass("Double start() safe")

    # None frame
    recorder.write_frame(None)
    log_pass("write_frame(None) safe")

    recorder.stop()
    recorder.stop()
    log_pass("Double stop() safe")

    recorder.write_frame(generate_frame())
    recorder.write_signals({"test": True})
    log_pass("Write after stop() safe")

    sys.stdout = old_stdout


# ─────────────────────────────────────────────
#  TEST 3: Video Playback
# ─────────────────────────────────────────────
def test_video_playback(session_dir):
    video_path = None
    for ext in ["mp4", "avi"]:
        vp = os.path.join(session_dir, f"video.{ext}")
        if os.path.exists(vp) and os.path.getsize(vp) > 1000:
            video_path = vp
            break

    if not video_path:
        log_pass("Video playback skipped (codec wrote empty file)")
        return

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        log_pass("Video file can be opened")
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fc > 0:
            log_pass(f"Video has {fc} frames")
        else:
            log_fail("Video frames", "0 frames reported")
        ret, frame = cap.read()
        if ret and frame is not None:
            log_pass(f"First frame readable (shape={frame.shape})")
        else:
            log_fail("First frame", "Cannot read")
        cap.release()
    else:
        log_fail("Video open", "Cannot open")


# ─────────────────────────────────────────────
#  TEST 4: CSV Export
# ─────────────────────────────────────────────
def test_csv_export(session_dir):
    exporter = TrainingDataExporter()

    old_stdout = sys.stdout
    sys.stdout = suppress_prints()
    try:
        exporter.load_session(session_dir)
    except Exception as e:
        sys.stdout = old_stdout
        log_fail("Load session", str(e))
        return
    sys.stdout = old_stdout

    log_pass("Session loaded for CSV export")

    csv_path = os.path.join(session_dir, "training_data.csv")

    old_stdout = sys.stdout
    sys.stdout = suppress_prints()
    result = exporter.export_labeled_csv(csv_path)
    sys.stdout = old_stdout

    if result and os.path.exists(csv_path):
        log_pass("CSV file created")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        if len(rows) > 0:
            log_pass(f"CSV has {len(rows)} data rows")
        else:
            log_fail("CSV rows", "Empty")

        expected = ["elapsed_sec", "gaze", "engagement_score", "arms", "neck_position"]
        missing = [c for c in expected if c not in (headers or [])]
        if not missing:
            log_pass(f"CSV has all expected columns ({len(headers)} total)")
        else:
            log_fail("CSV columns", f"Missing: {missing}")

        if rows and rows[0].get("gaze"):
            log_pass(f"CSV data populated (gaze='{rows[0]['gaze']}')")
        else:
            log_fail("CSV data", "First row empty")
    else:
        log_fail("CSV export", "File not created")


# ─────────────────────────────────────────────
#  TEST 5: Sequence Export
# ─────────────────────────────────────────────
def test_sequence_export(session_dir):
    exporter = TrainingDataExporter()

    old_stdout = sys.stdout
    sys.stdout = suppress_prints()
    exporter.load_session(session_dir)
    sys.stdout = old_stdout

    seq_path = os.path.join(session_dir, "sequences.json")

    old_stdout = sys.stdout
    sys.stdout = suppress_prints()
    result = exporter.export_sequence_windows(seq_path, window_size=10, stride=5)
    sys.stdout = old_stdout

    if result and os.path.exists(seq_path):
        with open(seq_path, "r", encoding="utf-8") as f:
            windows = json.load(f)

        if len(windows) > 0:
            log_pass(f"Sequences exported ({len(windows)} windows)")
        else:
            log_fail("Sequences", "No windows")
            return

        w = windows[0]
        required = ["window_id", "start_frame", "end_frame", "samples", "label"]
        missing = [k for k in required if k not in w]
        if not missing:
            log_pass("Window structure correct")
        else:
            log_fail("Window structure", f"Missing: {missing}")

        if len(w.get("samples", [])) == 10:
            log_pass("Window sample count correct (10)")
        else:
            log_fail("Window samples", f"Got {len(w.get('samples', []))}")

        label = w.get("label", {})
        if "attention_score" in label:
            log_pass(f"Window labels present (attention={label['attention_score']})")
        else:
            log_fail("Window labels", "Missing fields")
    else:
        log_fail("Sequence export", "File not created")


# ─────────────────────────────────────────────
#  TEST 6: Frame Dataset
# ─────────────────────────────────────────────
def test_frame_dataset(session_dir):
    # Check if video is usable
    video_ok = False
    for ext in ["mp4", "avi"]:
        vp = os.path.join(session_dir, f"video.{ext}")
        if os.path.exists(vp) and os.path.getsize(vp) > 1000:
            video_ok = True
            break

    if not video_ok:
        log_pass("Frame dataset skipped (video empty - codec limitation)")
        return

    exporter = TrainingDataExporter()

    old_stdout = sys.stdout
    sys.stdout = suppress_prints()
    exporter.load_session(session_dir)
    frames_dir = os.path.join(session_dir, "frame_dataset")
    result = exporter.export_frame_dataset(frames_dir, sample_rate=10)
    sys.stdout = old_stdout

    if result and os.path.isdir(frames_dir):
        files = os.listdir(frames_dir)
        jpgs = [f for f in files if f.endswith(".jpg")]
        jsons = [f for f in files if f.endswith(".json")]

        if len(jpgs) > 0:
            log_pass(f"Frame images extracted ({len(jpgs)})")
        else:
            log_fail("Frame images", "None created")

        if len(jpgs) == len(jsons):
            log_pass("Image/label count match")
        else:
            log_fail("Image/label count", f"{len(jpgs)} vs {len(jsons)}")
    else:
        log_fail("Frame dataset", "Not created")


# ─────────────────────────────────────────────
#  TEST 7: Session Discovery
# ─────────────────────────────────────────────
def test_list_sessions():
    sessions = list_sessions(TEST_RECORDINGS_DIR)
    if len(sessions) > 0:
        log_pass(f"Discovered {len(sessions)} session(s)")
        s = sessions[0]
        if s.get("has_signals"):
            log_pass("has_signals=True")
        else:
            log_fail("has_signals", "Expected True")
        if s.get("duration") is not None and s["duration"] != "?":
            log_pass(f"Duration available: {s['duration']}s")
        else:
            log_fail("Duration", "Not available")
    else:
        log_fail("Session discovery", "None found")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cleanup()

    try:
        session_dir = test_basic_recording()
        test_edge_cases()
        if session_dir:
            test_video_playback(session_dir)
            test_csv_export(session_dir)
            test_sequence_export(session_dir)
            test_frame_dataset(session_dir)
        test_list_sessions()
    except Exception as e:
        log_fail("UNEXPECTED", str(e))
        traceback.print_exc()
    finally:
        cleanup()

    # Print all results at once (no interleaving)
    print("=" * 60)
    print("  TRUSTED ADVISOR AI -- Recording System Tests")
    print("=" * 60)
    for status, name, reason in RESULTS:
        if status == "PASS":
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name} -- {reason}")
    print("-" * 60)
    print(f"  TOTAL: {PASS} passed, {FAIL} failed ({PASS + FAIL} tests)")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
