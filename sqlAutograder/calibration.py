"""
Grade calibration and curve module for SQL Autograder.

Applies post-processing score adjustments to correct for systematic LLM strictness bias.
Bias estimates are derived from observed Gemini and GPT-4.1-mini vs human grader deltas.

Usage:
    from calibration import ScoreCalibrator

    calibrator = ScoreCalibrator(mode="curve")
    adjusted = calibrator.calibrate_result(llm_result)

Modes:
    "curve"   — Additive per-question adjustment derived from observed bias (recommended)
    "scale"   — Multiplicative scaling to match human average per question
    "none"    — Passthrough, no adjustment applied
"""

from typing import Dict, Literal
import math


# ---------------------------------------------------------------------------
# Bias estimates (LLM avg - Human avg) observed across Gemini + GPT-4.1-mini
# These are conservative midpoints between the two models' biases.
#
#   Q4.1: Gemini -1.2, GPT -1.4  → midpoint -1.3
#   Q4.2: Gemini -0.7, GPT -0.9  → midpoint -0.8
#   Q4.3: Gemini -1.6, GPT -2.0  → midpoint -1.8
#   Q4.4: Gemini -2.7, GPT -3.1  → midpoint -2.9
#   Q4.5: Gemini -1.7, GPT -1.8  → midpoint -1.75
#
# Additive correction = -bias (we add back what the model subtracted too much)
# We apply 80% of the full correction to avoid over-shooting.
# ---------------------------------------------------------------------------
_ADDITIVE_CORRECTIONS = {
    'question_4_1': round(1.3 * 0.80, 2),   # +1.04
    'question_4_2': round(0.8 * 0.80, 2),   # +0.64
    'question_4_3': round(1.8 * 0.80, 2),   # +1.44
    'question_4_4': round(2.9 * 0.80, 2),   # +2.32
    'question_4_5': round(1.75 * 0.80, 2),  # +1.40
}

# Human grader averages (from statistics reports)
_HUMAN_AVERAGES = {
    'question_4_1': 8.9,
    'question_4_2': 9.2,
    'question_4_3': 8.8,
    'question_4_4': 7.7,
    'question_4_5': 8.8,
}

# LLM averages (midpoint of Gemini and GPT-4.1-mini observations)
_LLM_AVERAGES = {
    'question_4_1': (7.7 + 7.5) / 2,   # 7.60
    'question_4_2': (8.5 + 8.3) / 2,   # 8.40
    'question_4_3': (7.2 + 6.8) / 2,   # 7.00
    'question_4_4': (5.0 + 4.6) / 2,   # 4.80
    'question_4_5': (7.1 + 7.0) / 2,   # 7.05
}


class ScoreCalibrator:
    """
    Applies grade calibration to LLM grading results to correct for systematic
    over-strictness bias relative to human graders.

    Args:
        mode: Calibration strategy.
            "curve"  - Additive correction per question (recommended)
            "scale"  - Multiplicative scaling to match human average
            "none"   - No adjustment (passthrough)
        correction_factor: Fraction of observed bias to correct (default 0.8).
            1.0 = full correction to match human averages exactly.
            0.8 = 80% correction (conservative, avoids overshooting).
            0.0 = equivalent to mode="none".
    """

    VALID_MODES = {"curve", "scale", "none"}
    QUESTION_KEYS = [f'question_4_{i}' for i in [1, 2, 3, 4, 5]]

    def __init__(
        self,
        mode: Literal["curve", "scale", "none"] = "curve",
        correction_factor: float = 0.80,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        if not 0.0 <= correction_factor <= 1.0:
            raise ValueError(f"correction_factor must be between 0.0 and 1.0, got {correction_factor}")

        self.mode = mode
        self.correction_factor = correction_factor

        # Pre-compute per-question adjustments
        self._additive: Dict[str, float] = {
            k: round(v * (correction_factor / 0.80), 2)
            for k, v in _ADDITIVE_CORRECTIONS.items()
        }
        self._scale: Dict[str, float] = {}
        for k in self.QUESTION_KEYS:
            llm_avg = _LLM_AVERAGES.get(k, 1.0)
            human_avg = _HUMAN_AVERAGES.get(k, 10.0)
            if llm_avg > 0:
                raw_scale = human_avg / llm_avg
                # Blend: correction_factor towards human, rest stays at 1.0
                self._scale[k] = round(1.0 + (raw_scale - 1.0) * correction_factor, 4)
            else:
                self._scale[k] = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate_score(self, q_key: str, raw_score: float) -> float:
        """
        Calibrate a single question's raw LLM score.

        Args:
            q_key: Question key, e.g. 'question_4_1'
            raw_score: Raw LLM score (0-10). Negative scores (-1) pass through unchanged.

        Returns:
            Calibrated score clamped to [0, 10], or raw_score if it's negative.
        """
        if raw_score < 0:
            # Failed grading sentinel — leave untouched
            return raw_score

        if self.mode == "none":
            return raw_score

        if self.mode == "curve":
            adjusted = raw_score + self._additive.get(q_key, 0.0)
        elif self.mode == "scale":
            adjusted = raw_score * self._scale.get(q_key, 1.0)
        else:
            adjusted = raw_score

        # Clamp to valid range and round to whole number
        return int(round(min(10.0, max(0.0, adjusted))))

    def calibrate_result(self, llm_result: Dict) -> Dict:
        """
        Apply calibration to a full grading result dict (all 5 questions).

        Adds 'raw_score' field to each question entry preserving the original,
        and updates 'score' with the calibrated value.

        Args:
            llm_result: Dictionary from grader, keyed by question_4_x.

        Returns:
            New dictionary with calibrated scores. Original dict is not mutated.
        """
        if self.mode == "none":
            return llm_result

        calibrated = {}
        for q_key, q_data in llm_result.items():
            entry = dict(q_data)  # shallow copy
            raw = entry.get('score', 0.0)
            cal = self.calibrate_score(q_key, raw)
            entry['raw_score'] = raw
            entry['score'] = cal
            if cal != raw and cal >= 0:
                entry['calibration_applied'] = f"+{int(cal - raw)} ({self.mode} correction)"
            calibrated[q_key] = entry
        return calibrated

    def describe(self) -> str:
        """Return a human-readable summary of the calibration settings."""
        lines = [
            f"ScoreCalibrator(mode='{self.mode}', correction_factor={self.correction_factor})",
            "",
        ]
        if self.mode == "curve":
            lines.append("Additive corrections per question:")
            for k, v in self._additive.items():
                lines.append(f"  {k}: +{v:.2f} points")
        elif self.mode == "scale":
            lines.append("Multiplicative scale factors per question:")
            for k, v in self._scale.items():
                lines.append(f"  {k}: x{v:.4f}")
        else:
            lines.append("No calibration applied (passthrough mode).")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience: apply to a list of total scores (for reporting)
    # ------------------------------------------------------------------

    def calibrate_total(self, per_question_raw: Dict[str, float]) -> float:
        """
        Compute calibrated total score from a dict of raw per-question scores.

        Args:
            per_question_raw: {question_key: raw_score}

        Returns:
            Sum of calibrated per-question scores.
        """
        return sum(
            self.calibrate_score(k, v)
            for k, v in per_question_raw.items()
        )