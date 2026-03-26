"""
Attention Engine
================
Computes an overall attention score and state from aggregated metrics.

Output schema:
{
  "attention_score": 0.58,
  "attention_state": "HIGH | MEDIUM | LOW",
  "stability": "STABLE | UNSTABLE"
}
"""


class AttentionEngine:
    """Derives attention score from aggregated behavioural ratios."""

    # Weights for each negative-attention indicator
    WEIGHTS = {
        "off_screen_ratio": 0.35,
        "head_down_ratio": 0.25,
        "face_missing_ratio": 0.25,
        "slouch_ratio": 0.15,
    }

    # Thresholds
    HIGH_THRESHOLD = 0.70
    LOW_THRESHOLD = 0.40
    STABILITY_VARIANCE_LIMIT = 0.15

    def __init__(self):
        self._history: list[float] = []  # rolling scores for stability

    def compute(self, aggregated: dict) -> dict:
        """
        Parameters
        ----------
        aggregated : dict
            Output of AggregationEngine.aggregate()

        Returns
        -------
        dict with attention_score, attention_state, stability.
        """
        # Score = 1 - weighted sum of negative ratios (clamped 0..1)
        penalty = sum(
            self.WEIGHTS.get(key, 0) * aggregated.get(key, 0.0)
            for key in self.WEIGHTS
        )
        score = round(max(0.0, min(1.0, 1.0 - penalty)), 2)

        # State
        if score >= self.HIGH_THRESHOLD:
            state = "HIGH"
        elif score >= self.LOW_THRESHOLD:
            state = "MEDIUM"
        else:
            state = "LOW"

        # Stability – based on recent score variance
        self._history.append(score)
        if len(self._history) > 30:
            self._history = self._history[-30:]

        if len(self._history) >= 3:
            mean = sum(self._history) / len(self._history)
            variance = sum((s - mean) ** 2 for s in self._history) / len(self._history)
            stability = "STABLE" if variance < self.STABILITY_VARIANCE_LIMIT else "UNSTABLE"
        else:
            stability = "STABLE"

        return {
            "attention_score": score,
            "attention_state": state,
            "stability": stability,
        }

    def reset(self):
        self._history.clear()
