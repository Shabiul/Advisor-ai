"""
=============================================================
  Trusted Advisor AI — Training Data Exporter
  Converts recorded sessions into structured datasets
  for model training and fine-tuning.
=============================================================

Usage:
  python training_exporter.py --session <session_dir> --format csv
  python training_exporter.py --session <session_dir> --format frames --sample-rate 5
  python training_exporter.py --session <session_dir> --format sequences --window 30 --stride 10
  python training_exporter.py --list                 # list all available sessions

Formats:
  csv        — One CSV file with all behavioral signals per frame
  frames     — Extracted video frames + per-frame label JSON files
  sequences  — Sliding-window sequences for temporal/RNN models
"""

import os
import sys
import csv
import json
import argparse
import cv2
import time
from datetime import datetime


# ─────────────────────────────────────────────
#  TRAINING DATA EXPORTER
# ─────────────────────────────────────────────

class TrainingDataExporter:
    """
    Processes recorded Trusted Advisor AI sessions into structured
    datasets suitable for various machine learning training scenarios.
    """

    # Columns to extract for flat CSV export
    SIGNAL_COLUMNS = [
        "elapsed_sec", "frame_number",
        # Face signals
        "gaze", "eye_contact_score", "head_pose",
        "brow_score", "brow_label", "lip_score", "lip_label",
        "smile_genuine", "smile_label", "nodding", "head_shake",
        "blinks_per_minute", "micro_tension_score", "engagement_score",
        "face_detected",
        # Face sub-dict
        "cheeks", "lips_state", "lips_movement",
        # Body language
        "arms", "shoulder_alignment", "shoulder_energy", "shoulder_position",
        "neck_position", "neck_stability", "sitting_posture",
        # Attention (from report)
        "attention_score", "focus_level",
    ]

    def __init__(self):
        self._session_dir = None
        self._signals = []
        self._metadata = {}
        self._video_path = None

    def load_session(self, session_dir):
        """
        Load a recorded session from disk.

        Args:
            session_dir: Path to a session directory containing
                         signals.jsonl, video.mp4, and metadata.json
        """
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        self._session_dir = session_dir

        # Load signals
        signals_path = os.path.join(session_dir, "signals.jsonl")
        if not os.path.exists(signals_path):
            raise FileNotFoundError(f"Signal log not found: {signals_path}")

        self._signals = []
        with open(signals_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._signals.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[EXPORTER] Warning: Skipping malformed line {line_num}: {e}")

        # Load metadata
        metadata_path = os.path.join(session_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

        # Check for video
        for ext in ["mp4", "avi"]:
            vpath = os.path.join(session_dir, f"video.{ext}")
            if os.path.exists(vpath):
                self._video_path = vpath
                break

        print(f"[EXPORTER] Loaded session: {os.path.basename(session_dir)}")
        print(f"[EXPORTER]   Signals: {len(self._signals)} entries")
        print(f"[EXPORTER]   Video: {'found' if self._video_path else 'not found'}")
        print(f"[EXPORTER]   Duration: {self._metadata.get('duration_seconds', '?')}s")

    def export_labeled_csv(self, output_path=None):
        """
        Export all signal data as a flat CSV file.
        Each row = one sample (frame/signal snapshot).

        Args:
            output_path: Path for the output CSV. Defaults to
                         <session_dir>/training_data.csv
        """
        if not self._signals:
            print("[EXPORTER] No signals loaded. Call load_session() first.")
            return None

        if output_path is None:
            output_path = os.path.join(self._session_dir, "training_data.csv")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SIGNAL_COLUMNS)
            writer.writeheader()

            for entry in self._signals:
                row = self._flatten_entry(entry)
                writer.writerow(row)

        print(f"[EXPORTER] CSV exported: {output_path} ({len(self._signals)} rows)")
        return output_path

    def export_frame_dataset(self, output_dir=None, sample_rate=1):
        """
        Extract video frames at a given sample rate and pair each
        with a JSON label file containing behavioral signals.

        Args:
            output_dir: Directory for frame images + labels.
                        Defaults to <session_dir>/frame_dataset/
            sample_rate: Extract every Nth frame. Default=1 (all frames).

        Output:
            frame_dataset/
              frame_0001.jpg
              frame_0001.json    ← behavioral label for that frame
              frame_0002.jpg
              ...
        """
        if not self._video_path:
            print("[EXPORTER] No video file found. Cannot extract frames.")
            return None

        if output_dir is None:
            output_dir = os.path.join(self._session_dir, "frame_dataset")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            print(f"[EXPORTER] Cannot open video: {self._video_path}")
            return None

        # Build a lookup from frame_number to signal entry
        signal_lookup = {}
        for entry in self._signals:
            fn = entry.get("frame_number", -1)
            if fn >= 0:
                signal_lookup[fn] = entry

        frame_idx = 0
        exported = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                # Save frame image
                frame_name = f"frame_{exported:06d}"
                img_path = os.path.join(output_dir, f"{frame_name}.jpg")
                cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Save label
                label = self._flatten_entry(signal_lookup.get(frame_idx, {}))
                label_path = os.path.join(output_dir, f"{frame_name}.json")
                with open(label_path, "w", encoding="utf-8") as f:
                    json.dump(label, f, indent=2)

                exported += 1

            frame_idx += 1

        cap.release()
        print(f"[EXPORTER] Frame dataset exported: {output_dir} ({exported} frames)")
        return output_dir

    def export_sequence_windows(self, output_path=None, window_size=30, stride=10):
        """
        Export sliding-window sequences for temporal/RNN model training.
        Each window is a fixed-length sequence of signal snapshots.

        Args:
            output_path: Output JSON file path. Defaults to
                         <session_dir>/sequences.json
            window_size: Number of frames per sequence window.
            stride: Step size between windows.

        Output JSON structure:
            [
              {
                "window_id": 0,
                "start_frame": 0,
                "end_frame": 29,
                "start_time": 0.0,
                "end_time": 1.0,
                "samples": [ ... flattened signal dicts ... ],
                "label": { "attention_score": 0.85, ... }
              },
              ...
            ]
        """
        if not self._signals:
            print("[EXPORTER] No signals loaded. Call load_session() first.")
            return None

        if output_path is None:
            output_path = os.path.join(self._session_dir, "sequences.json")

        windows = []
        total = len(self._signals)

        for start in range(0, total - window_size + 1, stride):
            end = start + window_size
            window_entries = self._signals[start:end]

            samples = [self._flatten_entry(e) for e in window_entries]

            # Aggregate label for the window (use last entry's report as target)
            last_entry = window_entries[-1]
            report = last_entry.get("report", {})
            summary = report.get("summary", {})

            # Compute average engagement/attention over the window
            engagement_vals = [
                e.get("signals", {}).get("engagement_score", 5)
                for e in window_entries
            ]
            avg_engagement = round(sum(engagement_vals) / len(engagement_vals), 2)

            window_label = {
                "attention_score": summary.get("attention_score", 1.0),
                "focus_level": summary.get("focus_level", "HIGH"),
                "avg_engagement": avg_engagement,
            }

            windows.append({
                "window_id": len(windows),
                "start_frame": window_entries[0].get("frame_number", start),
                "end_frame": window_entries[-1].get("frame_number", end - 1),
                "start_time": window_entries[0].get("elapsed_sec", 0),
                "end_time": window_entries[-1].get("elapsed_sec", 0),
                "samples": samples,
                "label": window_label,
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(windows, f, indent=2)

        print(f"[EXPORTER] Sequence windows exported: {output_path} ({len(windows)} windows)")
        return output_path

    # ── Internal Helpers ──────────────────────────────────────────────

    def _flatten_entry(self, entry):
        """Flatten a signal log entry into a flat dict for CSV/labeling."""
        sig = entry.get("signals", {})
        report = entry.get("report", {})
        summary = report.get("summary", {})

        face = sig.get("face", {})
        lips = face.get("lips", {})
        bl = sig.get("body_language", {})
        shoulders = bl.get("shoulders", {})
        neck = bl.get("neck", {})

        return {
            "elapsed_sec": entry.get("elapsed_sec", 0),
            "frame_number": entry.get("frame_number", 0),
            # Face signals
            "gaze": sig.get("gaze", ""),
            "eye_contact_score": sig.get("eye_contact_score", 0),
            "head_pose": sig.get("head_pose", ""),
            "brow_score": sig.get("brow_score", 0),
            "brow_label": sig.get("brow_label", ""),
            "lip_score": sig.get("lip_score", 0),
            "lip_label": sig.get("lip_label", ""),
            "smile_genuine": sig.get("smile_genuine", False),
            "smile_label": sig.get("smile_label", ""),
            "nodding": sig.get("nodding", False),
            "head_shake": sig.get("head_shake", False),
            "blinks_per_minute": sig.get("blinks_per_minute", 0),
            "micro_tension_score": sig.get("micro_tension_score", 0),
            "engagement_score": sig.get("engagement_score", 0),
            "face_detected": sig.get("face_detected", False),
            # Face sub-dict
            "cheeks": face.get("cheeks", ""),
            "lips_state": lips.get("state", "") if isinstance(lips, dict) else "",
            "lips_movement": lips.get("movement", "") if isinstance(lips, dict) else "",
            # Body language
            "arms": bl.get("arms", ""),
            "shoulder_alignment": shoulders.get("alignment", "") if isinstance(shoulders, dict) else "",
            "shoulder_energy": shoulders.get("energy", "") if isinstance(shoulders, dict) else "",
            "shoulder_position": shoulders.get("position", "") if isinstance(shoulders, dict) else "",
            "neck_position": neck.get("position", "") if isinstance(neck, dict) else "",
            "neck_stability": neck.get("stability", "") if isinstance(neck, dict) else "",
            "sitting_posture": bl.get("sitting_posture", ""),
            # From report
            "attention_score": summary.get("attention_score", ""),
            "focus_level": summary.get("focus_level", ""),
        }


# ─────────────────────────────────────────────
#  SESSION DISCOVERY
# ─────────────────────────────────────────────

def find_recordings_dir():
    """Find the recordings directory relative to this script."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "recordings")


def list_sessions(recordings_dir=None):
    """List all available recorded sessions."""
    if recordings_dir is None:
        recordings_dir = find_recordings_dir()

    if not os.path.isdir(recordings_dir):
        print(f"[EXPORTER] No recordings directory found at: {recordings_dir}")
        return []

    sessions = []
    for name in sorted(os.listdir(recordings_dir)):
        session_path = os.path.join(recordings_dir, name)
        if not os.path.isdir(session_path):
            continue

        # Check for metadata
        meta_path = os.path.join(session_path, "metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                pass

        # Check for files
        has_video = any(
            os.path.exists(os.path.join(session_path, f"video.{ext}"))
            for ext in ["mp4", "avi"]
        )
        has_signals = os.path.exists(os.path.join(session_path, "signals.jsonl"))

        sessions.append({
            "name": name,
            "path": session_path,
            "has_video": has_video,
            "has_signals": has_signals,
            "duration": meta.get("duration_seconds", "?"),
            "frames": meta.get("total_frames", "?"),
            "start_time": meta.get("start_time", "?"),
        })

    return sessions


# ─────────────────────────────────────────────
#  CLI INTERFACE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trusted Advisor AI — Training Data Exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training_exporter.py --list
  python training_exporter.py --session recordings/session_2026-03-31_13-45-00 --format csv
  python training_exporter.py --session recordings/session_2026-03-31_13-45-00 --format frames --sample-rate 5
  python training_exporter.py --session recordings/session_2026-03-31_13-45-00 --format sequences --window 30 --stride 10
        """,
    )
    parser.add_argument("--list", action="store_true", help="List all available recorded sessions")
    parser.add_argument("--session", type=str, help="Path to session directory")
    parser.add_argument(
        "--format", type=str, choices=["csv", "frames", "sequences"],
        default="csv", help="Export format (default: csv)",
    )
    parser.add_argument("--output", type=str, help="Output path (file or directory)")
    parser.add_argument("--sample-rate", type=int, default=1, help="Frame sampling rate for 'frames' format")
    parser.add_argument("--window", type=int, default=30, help="Window size for 'sequences' format")
    parser.add_argument("--stride", type=int, default=10, help="Stride for 'sequences' format")

    args = parser.parse_args()

    if args.list:
        sessions = list_sessions()
        if not sessions:
            print("No recorded sessions found.")
            return

        print(f"\n{'='*70}")
        print(f"  RECORDED SESSIONS ({len(sessions)} found)")
        print(f"{'='*70}")
        for s in sessions:
            vid = "✓" if s["has_video"] else "✗"
            sig = "✓" if s["has_signals"] else "✗"
            print(f"  {s['name']}")
            print(f"    Duration: {s['duration']}s | Frames: {s['frames']} | Video: {vid} | Signals: {sig}")
            print(f"    Started:  {s['start_time']}")
            print(f"    Path:     {s['path']}")
            print()
        return

    if not args.session:
        parser.error("--session is required (or use --list to see available sessions)")

    # Resolve session path
    session_dir = args.session
    if not os.path.isabs(session_dir):
        # Try relative to recordings dir
        rec_dir = find_recordings_dir()
        candidate = os.path.join(rec_dir, session_dir)
        if os.path.isdir(candidate):
            session_dir = candidate

    exporter = TrainingDataExporter()
    exporter.load_session(session_dir)

    if args.format == "csv":
        exporter.export_labeled_csv(args.output)
    elif args.format == "frames":
        exporter.export_frame_dataset(args.output, sample_rate=args.sample_rate)
    elif args.format == "sequences":
        exporter.export_sequence_windows(args.output, window_size=args.window, stride=args.stride)

    print("\n[EXPORTER] Done!")


if __name__ == "__main__":
    main()
