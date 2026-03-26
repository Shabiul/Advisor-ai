"""
Report Generator (CRITICAL)
============================
Produces the final structured report consolidating attention, aggregation,
and body analysis data.

Output schema:
{
  "summary": { "attention_score": 0.58, "focus_level": "LOW" },
  "metrics": { "off_screen_time": "32%", "head_down_time": "18%" },
  "alerts": [ {"type": "LOOK_AWAY", "count": 12, "duration": "40s"} ],
  "body_analysis": { "posture": "SLOUCHED", "neck": "FORWARD_HEAD", "shoulders": "DROPPED" },
  "recommendation": "REVIEW_REQUIRED | ACCEPTABLE | GOOD"
}
"""


class ReportGenerator:
    """Assembles all analytics outputs into a single human-reviewable report."""

    # Alerts are generated only for events exceeding these count thresholds
    ALERT_THRESHOLDS = {
        "LOOK_AWAY": 5,
        "LOOK_DOWN": 4,
        "FACE_MISSING": 3,
        "SLOUCH": 4,
        "SHOULDER_DROP": 3,
        "ARMS_CROSSED": 5,
    }

    def generate(
        self,
        attention: dict,
        aggregated: dict,
        latest_pose: dict | None,
    ) -> dict:
        """
        Parameters
        ----------
        attention   : output of AttentionEngine.compute()
        aggregated  : output of AggregationEngine.aggregate()
        latest_pose : most recent pose signals (for body_analysis snapshot)

        Returns
        -------
        Structured report dict.
        """
        summary = {
            "attention_score": attention["attention_score"],
            "focus_level": attention["attention_state"],
            "stability": attention["stability"],
        }

        metrics = {
            "off_screen_time": f"{int(aggregated['off_screen_ratio'] * 100)}%",
            "head_down_time": f"{int(aggregated['head_down_ratio'] * 100)}%",
            "face_missing_time": f"{int(aggregated['face_missing_ratio'] * 100)}%",
            "slouch_time": f"{int(aggregated['slouch_ratio'] * 100)}%",
        }

        # Alerts for events exceeding thresholds
        alerts = []
        for etype, count in aggregated.get("event_counts", {}).items():
            threshold = self.ALERT_THRESHOLDS.get(etype, 5)
            if count >= threshold:
                duration = aggregated.get("event_durations", {}).get(etype, 0)
                alerts.append({
                    "type": etype,
                    "count": count,
                    "duration": f"{duration}s",
                })

        # Body analysis from latest pose
        body_analysis = self._extract_body(latest_pose)

        # Recommendation
        score = attention["attention_score"]
        if score >= 0.70 and len(alerts) == 0:
            recommendation = "GOOD"
        elif score >= 0.40:
            recommendation = "ACCEPTABLE"
        else:
            recommendation = "REVIEW_REQUIRED"

        return {
            "summary": summary,
            "metrics": metrics,
            "alerts": alerts,
            "body_analysis": body_analysis,
            "recommendation": recommendation,
        }

    @staticmethod
    def _extract_body(pose: dict | None) -> dict:
        if pose is None:
            return {
                "posture": "UNKNOWN",
                "neck": "UNKNOWN",
                "shoulders": "UNKNOWN",
            }
        shoulders = pose.get("shoulders", {})
        return {
            "posture": pose.get("sitting_posture", "UNKNOWN"),
            "neck": pose.get("neck", "UNKNOWN"),
            "shoulders": shoulders.get("energy", "UNKNOWN"),
        }
