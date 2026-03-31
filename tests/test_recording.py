"""
=============================================================
  Trusted Advisor AI — Recording System Tests
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


def log_pass(name):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}")


def log_fail(name, reason=""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name} -- {reason}")


def cleanup():
    """Remove test recordings directory."""
    if os.path.exists(TEST_RECORDINGS_DIR):
        shutil.rmtree(TEST_RECORDINGS_DIR)


def generate_synthetic_frame(width=640, height=480, frame_num=0):
    """Generate a synthetic BGR frame with some visual content."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Gradient background
    for y in range(height):
        frame[y, :, 0] = int(255 * y / height)  # Blue gradient
        frame[y, :, 2] = int(255 * (1 - y / height))  # Red gradient
    # Moving circle to verify frame ordering
    cx = int((frame_num * 10) % width)
    cy = height // 2
    cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
    cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def generate_synthetic_signal(frame_num, elapsed):
    """Generate a realistic-looking signal dictionary."""
    return {
        "gaze": "CENTER" if frame_num % 5 != 0 else "LOOKING LEFT",
        "eye_contact_score": round(0.5 + 0.3 * np.sin(frame_num * 0.1), 2),
        "head_pose": "FACING FORWARD" if frame_num % 7 != 0 else "TURNED RIGHT",
        "brow_score": round(0.5 + 0.1 * np.cos(frame_num * 0.05), 2),
        "brow_label": "NEUTRAL",
        "lip_score": 0.5,
        "lip_label": "RELAXED",
        "smile_genuine": frame_num % 10 == 0,
        "smile_label": "GENUINE" if frame_num % 10 == 0 else "NONE",
        "nodding": frame_num % 15 == 0,
        "head_shake": False,
        "blinks_per_minute": 15 + int(5 * np.sin(frame_num * 0.02)),
        "micro_tension_score": int(2 * abs(np.sin(frame_num * 0.03))),
        "engagement_score": 5 + int(3 * np.sin(frame_num * 0.01)),
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


def generate_synthetic_report(sig):
    """Generate a minimal cumulative report."""
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
#  TEST 1: SessionRecorder — Basic Recording
# ─────────────────────────────────────────────
def test_session_recorder_basic():
    print("\n[TEST 1] SessionRecorder -- Basic Recording")

    recorder = SessionRecorder(base_dir=TEST_RECORDINGS_DIR, fps=30)

    # Test initial state
    if not recorder.is_recording:
        log_pass("Initial state is not recording")
    else:
        log_fail("Initial state check", "Should not be recording before start()")

    # Start recording
    recorder.start(frame_width=640, frame_height=480)

    if recorder.is_recording:
        log_pass("Recording started successfully")
    else:
        log_fail("Recording start", "is_recording should be True after start()")

    if recorder.session_dir and os.path.isdir(recorder.session_dir):
        log_pass(f"Session directory created: {os.path.basename(recorder.session_dir)}")
    else:
        log_fail("Session directory", "Directory not created")

    # Write 60 synthetic frames (~2 seconds at 30fps)
    num_frames = 60
    for i in range(num_frames):
        frame = generate_synthetic_frame(640, 480, i)
        recorder.write_frame(frame)

        sig = generate_synthetic_signal(i, i / 30.0)
        report = generate_synthetic_report(sig) if i % 15 == 0 else None
        recorder.write_signals(sig, report)

    if recorder.frame_count == num_frames:
        log_pass(f"Frame count matches: {recorder.frame_count}/{num_frames}")
    else:
        log_fail("Frame count", f"Expected {num_frames}, got {recorder.frame_count}")

    # Stop recording
    session_dir = recorder.session_dir
    recorder.stop()

    if not recorder.is_recording:
        log_pass("Recording stopped successfully")
    else:
        log_fail("Recording stop", "is_recording should be False after stop()")

    # Verify output files
    video_path = os.path.join(session_dir, "video.mp4")
    video_avi = os.path.join(session_dir, "video.avi")
    has_video = os.path.exists(video_path) or os.path.exists(video_avi)
    if has_video:
        vp = video_path if os.path.exists(video_path) else video_avi
        vsize = os.path.getsize(vp)
        log_pass(f"Video file created ({vsize:,} bytes)")
    else:
        log_fail("Video file", "Neither video.mp4 nor video.avi found")

    signals_path = os.path.join(session_dir, "signals.jsonl")
    if os.path.exists(signals_path):
        with open(signals_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        log_pass(f"Signals file created ({len(lines)} entries)")

        # Verify each line is valid JSON
        all_valid = True
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
                if "timestamp" not in obj or "signals" not in obj:
                    all_valid = False
                    break
            except json.JSONDecodeError:
                all_valid = False
                break

        if all_valid:
            log_pass("All signal entries are valid JSON with required fields")
        else:
            log_fail("Signal JSON validity", f"Invalid entry at line {i + 1}")
    else:
        log_fail("Signals file", "signals.jsonl not found")

    metadata_path = os.path.join(session_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        required_fields = ["session_id", "start_time", "end_time", "duration_seconds",
                           "total_frames", "total_signals", "fps"]
        missing = [f for f in required_fields if f not in meta]
        if not missing:
            log_pass(f"Metadata file valid (duration={meta['duration_seconds']}s, " +
                     f"frames={meta['total_frames']}, signals={meta['total_signals']})")
        else:
            log_fail("Metadata fields", f"Missing: {missing}")
    else:
        log_fail("Metadata file", "metadata.json not found")

    return session_dir


# ─────────────────────────────────────────────
#  TEST 2: SessionRecorder — Idempotency
# ─────────────────────────────────────────────
def test_session_recorder_idempotency():
    print("\n[TEST 2] SessionRecorder -- Idempotency & Edge Cases")

    recorder = SessionRecorder(base_dir=TEST_RECORDINGS_DIR, fps=30)

    # Double start should not crash
    recorder.start(frame_width=640, frame_height=480)
    recorder.start(frame_width=640, frame_height=480)  # should be no-op
    log_pass("Double start() does not crash")

    # Write with no frame should not crash
    recorder.write_frame(None)
    log_pass("write_frame(None) does not crash")

    # Stop
    recorder.stop()

    # Double stop should not crash
    recorder.stop()
    log_pass("Double stop() does not crash")

    # Write after stop should not crash
    frame = generate_synthetic_frame(640, 480, 0)
    recorder.write_frame(frame)
    recorder.write_signals({"test": True})
    log_pass("Writing after stop() does not crash")


# ─────────────────────────────────────────────
#  TEST 3: SessionRecorder — Video Playback
# ─────────────────────────────────────────────
def test_video_playback(session_dir):
    print("\n[TEST 3] Video Playback Verification")

    video_path = os.path.join(session_dir, "video.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(session_dir, "video.avi")
    if not os.path.exists(video_path):
        log_fail("Video playback", "No video file found")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_fail("Video open", "Cannot open recorded video")
        return

    log_pass("Recorded video can be opened by OpenCV")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count > 0:
        log_pass(f"Video has {frame_count} frames, {fps:.1f} FPS, {width}x{height}")
    else:
        log_fail("Video frames", "Video reports 0 frames")

    # Read first frame
    ret, frame = cap.read()
    if ret and frame is not None and frame.shape[0] > 0:
        log_pass(f"First frame readable (shape={frame.shape})")
    else:
        log_fail("First frame", "Cannot read first frame")

    cap.release()


# ─────────────────────────────────────────────
#  TEST 4: TrainingDataExporter — CSV Export
# ─────────────────────────────────────────────
def test_csv_export(session_dir):
    print("\n[TEST 4] TrainingDataExporter -- CSV Export")

    exporter = TrainingDataExporter()

    try:
        exporter.load_session(session_dir)
        log_pass("Session loaded successfully")
    except Exception as e:
        log_fail("Load session", str(e))
        return

    csv_path = os.path.join(session_dir, "test_training_data.csv")
    result = exporter.export_labeled_csv(csv_path)

    if result and os.path.exists(csv_path):
        log_pass(f"CSV file created: {os.path.basename(csv_path)}")

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            headers = reader.fieldnames

        if len(rows) > 0:
            log_pass(f"CSV has {len(rows)} rows")
        else:
            log_fail("CSV rows", "No data rows in CSV")

        expected_cols = ["elapsed_sec", "gaze", "engagement_score", "attention_score",
                         "arms", "neck_position"]
        missing_cols = [c for c in expected_cols if c not in headers]
        if not missing_cols:
            log_pass(f"CSV has all expected columns ({len(headers)} total)")
        else:
            log_fail("CSV columns", f"Missing: {missing_cols}")

        # Verify data values are populated
        first_row = rows[0]
        if first_row.get("gaze"):
            log_pass(f"CSV data populated (gaze='{first_row['gaze']}')")
        else:
            log_fail("CSV data", "First row gaze is empty")
    else:
        log_fail("CSV export", "File not created")


# ─────────────────────────────────────────────
#  TEST 5: TrainingDataExporter — Sequence Export
# ─────────────────────────────────────────────
def test_sequence_export(session_dir):
    print("\n[TEST 5] TrainingDataExporter -- Sequence Windows Export")

    exporter = TrainingDataExporter()
    exporter.load_session(session_dir)

    seq_path = os.path.join(session_dir, "test_sequences.json")
    result = exporter.export_sequence_windows(
        output_path=seq_path, window_size=10, stride=5
    )

    if result and os.path.exists(seq_path):
        with open(seq_path, "r") as f:
            windows = json.load(f)

        if len(windows) > 0:
            log_pass(f"Sequence file created with {len(windows)} windows")
        else:
            log_fail("Sequence count", "No windows generated")
            return

        w = windows[0]
        required = ["window_id", "start_frame", "end_frame", "samples", "label"]
        missing = [k for k in required if k not in w]
        if not missing:
            log_pass("Window structure correct (has all required fields)")
        else:
            log_fail("Window structure", f"Missing: {missing}")

        if len(w["samples"]) == 10:
            log_pass(f"Window sample count correct ({len(w['samples'])} samples)")
        else:
            log_fail("Window samples", f"Expected 10, got {len(w['samples'])}")

        label = w.get("label", {})
        if "attention_score" in label and "avg_engagement" in label:
            log_pass(f"Window labels present (attention={label['attention_score']}, " +
                     f"engagement={label['avg_engagement']})")
        else:
            log_fail("Window labels", f"Missing expected label fields")
    else:
        log_fail("Sequence export", "File not created")


# ─────────────────────────────────────────────
#  TEST 6: TrainingDataExporter — Frame Dataset
# ─────────────────────────────────────────────
def test_frame_dataset_export(session_dir):
    print("\n[TEST 6] TrainingDataExporter -- Frame Dataset Export")

    exporter = TrainingDataExporter()
    exporter.load_session(session_dir)

    frames_dir = os.path.join(session_dir, "test_frame_dataset")
    result = exporter.export_frame_dataset(
        output_dir=frames_dir, sample_rate=10
    )

    if result and os.path.isdir(frames_dir):
        files = os.listdir(frames_dir)
        jpg_files = [f for f in files if f.endswith(".jpg")]
        json_files = [f for f in files if f.endswith(".json")]

        if len(jpg_files) > 0:
            log_pass(f"Frame images extracted ({len(jpg_files)} files)")
        else:
            log_fail("Frame images", "No JPG files created")

        if len(json_files) > 0:
            log_pass(f"Frame labels created ({len(json_files)} files)")
        else:
            log_fail("Frame labels", "No JSON label files created")

        if len(jpg_files) == len(json_files):
            log_pass("Image count matches label count")
        else:
            log_fail("Image-label count", f"{len(jpg_files)} images vs {len(json_files)} labels")

        # Verify a label file
        if json_files:
            label_path = os.path.join(frames_dir, sorted(json_files)[0])
            with open(label_path, "r") as f:
                label = json.load(f)
            if "gaze" in label or "elapsed_sec" in label:
                log_pass("Label file contains expected signal fields")
            else:
                log_fail("Label content", "Missing expected fields")
    else:
        log_fail("Frame dataset export", "Directory not created")


# ─────────────────────────────────────────────
#  TEST 7: list_sessions()
# ─────────────────────────────────────────────
def test_list_sessions():
    print("\n[TEST 7] list_sessions() Discovery")

    sessions = list_sessions(TEST_RECORDINGS_DIR)
    if len(sessions) > 0:
        log_pass(f"Found {len(sessions)} session(s)")

        s = sessions[0]
        if s.get("has_signals"):
            log_pass("Session reports has_signals=True")
        else:
            log_fail("Session signals flag", "Expected has_signals=True")

        if s.get("has_video"):
            log_pass("Session reports has_video=True")
        else:
            log_fail("Session video flag", "Expected has_video=True")

        if s.get("duration") and s["duration"] != "?":
            log_pass(f"Session duration available: {s['duration']}s")
        else:
            log_fail("Session duration", "Duration not available")
    else:
        log_fail("Session discovery", "No sessions found")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TRUSTED ADVISOR AI — Recording System Tests")
    print("=" * 60)

    cleanup()

    try:
        session_dir = test_session_recorder_basic()
        test_session_recorder_idempotency()
        if session_dir:
            test_video_playback(session_dir)
            test_csv_export(session_dir)
            test_sequence_export(session_dir)
            test_frame_dataset_export(session_dir)
        test_list_sessions()
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        FAIL += 1
    finally:
        # Clean up test data
        cleanup()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed ({PASS + FAIL} total)")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
