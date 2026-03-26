"""
Pose Analyzer Module
====================
Extracts upper-body posture signals from a video frame using MediaPipe Pose.

Output schema (per frame):
{
  "shoulders": {
    "alignment": "STRAIGHT | TILTED",
    "energy": "ACTIVE | DROPPED",
    "position": "FORWARD | NEUTRAL"
  },
  "neck": "STRAIGHT | FORWARD_HEAD | DOWN",
  "arms": "OPEN | CROSSED | RELAXED",
  "sitting_posture": "UPRIGHT | LEAN_FORWARD | LEAN_BACK | SLOUCHED"
}
"""

import math
import mediapipe as mp

mp_pose = mp.solutions.pose

# MediaPipe Pose landmark indices
NOSE = mp_pose.PoseLandmark.NOSE
LEFT_EAR = mp_pose.PoseLandmark.LEFT_EAR
RIGHT_EAR = mp_pose.PoseLandmark.RIGHT_EAR
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW
RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP


def _angle_between(a, b, c):
    """Return angle at vertex b (in degrees) formed by points a-b-c."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_a = math.sqrt(ba[0] ** 2 + ba[1] ** 2) + 1e-6
    mag_c = math.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-6
    cos_angle = max(-1, min(1, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_angle))


class PoseAnalyzer:
    """Stateless upper-body pose analyzer."""

    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame_rgb) -> dict | None:
        """Return pose-signal dict or None when body not detected."""
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark

        shoulders = self._compute_shoulders(lm)
        neck = self._compute_neck(lm)
        arms = self._compute_arms(lm)
        sitting_posture = self._compute_sitting_posture(lm)

        return {
            "shoulders": shoulders,
            "neck": neck,
            "arms": arms,
            "sitting_posture": sitting_posture,
        }

    # ── private helpers ───────────────────────────────────────────────

    @staticmethod
    def _compute_shoulders(lm) -> dict:
        ls = lm[LEFT_SHOULDER]
        rs = lm[RIGHT_SHOULDER]

        # Alignment – compare y-coordinates
        y_diff = abs(ls.y - rs.y)
        alignment = "TILTED" if y_diff > 0.04 else "STRAIGHT"

        # Energy – dropped shoulders have higher y (lower in frame)
        mid_shoulder_y = (ls.y + rs.y) / 2
        nose_y = lm[NOSE].y
        drop_ratio = mid_shoulder_y - nose_y
        energy = "DROPPED" if drop_ratio > 0.28 else "ACTIVE"

        # Position – check z-depth; forward shoulders have lower z
        avg_z = (ls.z + rs.z) / 2
        position = "FORWARD" if avg_z < -0.06 else "NEUTRAL"

        return {"alignment": alignment, "energy": energy, "position": position}

    @staticmethod
    def _compute_neck(lm) -> str:
        nose = lm[NOSE]
        mid_shoulder_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2
        mid_shoulder_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2

        # Forward head – nose is significantly more forward (lower z) than shoulders
        avg_shoulder_z = (lm[LEFT_SHOULDER].z + lm[RIGHT_SHOULDER].z) / 2
        if nose.z < avg_shoulder_z - 0.08:
            return "FORWARD_HEAD"

        # Head down – nose y is close to or below mid-shoulder y
        if nose.y > mid_shoulder_y - 0.08:
            return "DOWN"

        return "STRAIGHT"

    @staticmethod
    def _compute_arms(lm) -> str:
        lw = lm[LEFT_WRIST]
        rw = lm[RIGHT_WRIST]
        le = lm[LEFT_ELBOW]
        re = lm[RIGHT_ELBOW]

        # Crossed – wrists are on opposite sides of the body midline
        mid_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2
        left_crossed = lw.x > mid_x  # left wrist on right side
        right_crossed = rw.x < mid_x  # right wrist on left side

        if left_crossed and right_crossed:
            return "CROSSED"

        # Open – elbows far from body
        elbow_spread = abs(le.x - re.x)
        shoulder_spread = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
        if elbow_spread > shoulder_spread * 1.4:
            return "OPEN"

        return "RELAXED"

    @staticmethod
    def _compute_sitting_posture(lm) -> str:
        """Classify overall sitting posture from spine curvature proxy."""
        nose = lm[NOSE]
        mid_shoulder_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2
        mid_shoulder_z = (lm[LEFT_SHOULDER].z + lm[RIGHT_SHOULDER].z) / 2

        # Use hip landmarks if visible
        lh = lm[LEFT_HIP]
        rh = lm[RIGHT_HIP]
        hip_vis = (lh.visibility + rh.visibility) / 2

        if hip_vis < 0.4:
            # Hips not visible – use shoulder-nose relationship only
            if nose.z < mid_shoulder_z - 0.10:
                return "LEAN_FORWARD"
            return "UPRIGHT"

        mid_hip_y = (lh.y + rh.y) / 2
        torso_length = mid_hip_y - mid_shoulder_y

        # Lean forward – shoulder z much lower than hip z
        mid_hip_z = (lh.z + rh.z) / 2
        if mid_shoulder_z < mid_hip_z - 0.08:
            return "LEAN_FORWARD"

        # Lean back – shoulders behind hips
        if mid_shoulder_z > mid_hip_z + 0.08:
            return "LEAN_BACK"

        # Slouch – short torso length (shoulders close to hips in y)
        if torso_length < 0.15:
            return "SLOUCHED"

        return "UPRIGHT"
