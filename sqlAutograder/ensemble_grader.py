"""
Ensemble grading module — combines Gemini, OpenAI o4-mini, and Claude via median vote.

Strategy:
  - Runs all three graders concurrently for each student
  - Per question: takes the integer median of the three scores
  - Flags any question where models disagree by >= 3 points (needs_review = True)
  - Feedback is taken from the model whose score is closest to the median
  - Falls back gracefully: if one model fails, uses mean of the remaining two

Usage:
    from sqlAutograder import EnsembleGrader
    grader = EnsembleGrader()
    result, error = grader.grade_student_submission(student_queries)
"""

import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .config import get_gemini_config, get_openai_config, get_claude_config
from .grader import GeminiGrader
from .openai_grader import OpenAIGrader
from .claude_grader import ClaudeGrader

# How far apart scores can be before flagging for human review
_DISAGREEMENT_THRESHOLD = 3

ENSEMBLE_MODELS = ["gemini-2.5-flash", "o4-mini", "claude-sonnet-4-6"]


class EnsembleGrader:
    """
    Runs Gemini, OpenAI o4-mini, and Claude concurrently and combines their
    scores using a per-question median vote.

    Args:
        calibration: Whether individual graders apply calibration internally.
                     The ensemble itself does NOT apply an additional calibration
                     layer — the median vote across calibrated scores is the output.
    """

    def __init__(self):
        gemini_config  = get_gemini_config()
        openai_config  = get_openai_config(model_name="o4-mini")
        claude_config  = get_claude_config(model_name="claude-sonnet-4-6")

        self.graders = {
            "gemini": GeminiGrader(gemini_config),
            "o4-mini": OpenAIGrader(openai_config),
            "claude": ClaudeGrader(claude_config),
        }

        # Expose a .config stub so main.py's default_workers check doesn't crash
        self.config = _EnsembleConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grade_student_submission(
        self,
        student_queries: Dict[str, str],
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Grade one student by running all three models concurrently, then
        combining per-question scores via median vote.

        Returns:
            Tuple of (combined_result, error_message)
            combined_result keys: question_4_1 … question_4_5, each with:
                score, raw_scores, feedback, deduction_details, needs_review,
                model_scores, disagreement
        """
        raw_results: Dict[str, Tuple[Optional[Dict], Optional[str]]] = {}

        # Run all three graders in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_name = {
                executor.submit(grader.grade_student_submission, student_queries): name
                for name, grader in self.graders.items()
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    raw_results[name] = future.result()
                except Exception as e:
                    raw_results[name] = (None, str(e))

        return self._combine(raw_results)

    def grade_batch(
        self,
        submissions: List[Tuple[str, Dict[str, str]]],
        max_workers: int = 3,
    ) -> Dict[str, Tuple[Optional[Dict], Optional[str]]]:
        """
        Grade multiple students.  Each student triggers 3 concurrent model calls
        internally, so keep max_workers modest to avoid rate limits.

        Args:
            submissions: List of (student_id, queries_dict) tuples
            max_workers: Parallel students (default 3 — each spawns 3 model calls)

        Returns:
            Dict mapping student_id → (combined_result, error_message)
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.grade_student_submission, queries): student_id
                for student_id, queries in submissions
            }
            for future in as_completed(future_to_id):
                student_id = future_to_id[future]
                try:
                    results[student_id] = future.result()
                except Exception as e:
                    results[student_id] = (None, str(e))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _combine(
        self,
        raw_results: Dict[str, Tuple[Optional[Dict], Optional[str]]],
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Combine per-model results into a single median-vote result dict."""

        questions = ["question_4_1", "question_4_2", "question_4_3",
                     "question_4_4", "question_4_5"]

        # Separate successful vs failed model results
        successful: Dict[str, Dict] = {}
        errors: Dict[str, str] = {}
        for name, (result, error) in raw_results.items():
            if result is not None:
                successful[name] = result
            else:
                errors[name] = error or "unknown error"

        if len(successful) == 0:
            error_summary = "; ".join(f"{n}: {e}" for n, e in errors.items())
            return None, f"All models failed: {error_summary}"

        if len(successful) == 1:
            only_name = next(iter(successful))
            # Still usable but flag every question for review
            result = successful[only_name]
            for q in questions:
                if q in result:
                    result[q]['needs_review'] = True
                    result[q]['ensemble_note'] = (
                        f"Only {only_name} succeeded; "
                        + ", ".join(f"{n}: {e}" for n, e in errors.items())
                    )
            return result, None

        combined: Dict = {}

        for q in questions:
            # Collect scores from every model that returned this question
            model_scores: Dict[str, int] = {}
            for name, result in successful.items():
                q_data = result.get(q, {})
                score = q_data.get('score', -1)
                if score >= 0:
                    model_scores[name] = int(round(score))

            if not model_scores:
                combined[q] = {
                    'score': -1,
                    'deduction_details': 'All models failed this question',
                    'feedback': 'Requires manual review',
                    'needs_review': True,
                    'model_scores': {},
                    'disagreement': 0,
                }
                continue

            scores = list(model_scores.values())
            median_score = int(round(statistics.median(scores)))
            disagreement = max(scores) - min(scores)
            needs_review = disagreement >= _DISAGREEMENT_THRESHOLD

            # Pick feedback from the model closest to the median
            best_model = min(
                model_scores,
                key=lambda n: abs(model_scores[n] - median_score)
            )
            best_q_data = successful[best_model].get(q, {})

            combined[q] = {
                'score': median_score,
                'raw_scores': model_scores,           # per-model breakdown
                'deduction_details': best_q_data.get('deduction_details', ''),
                'feedback': (
                    f"[Ensemble median={median_score}, "
                    f"models={model_scores}] "
                    + best_q_data.get('feedback', '')
                ),
                'needs_review': needs_review,
                'model_scores': model_scores,
                'disagreement': disagreement,
            }

        return combined, None

    @staticmethod
    def create_failed_result(
        question_numbers: List[str],
        error_message: str = "Grading failed",
    ) -> Dict:
        result = {}
        for q_num in question_numbers:
            q_key = f'question_{q_num.replace(".", "_")}'
            result[q_key] = {
                'score': -1,
                'deduction_details': error_message,
                'feedback': 'Automatic grading failed - requires manual review',
                'needs_review': True,
            }
        return result


class _EnsembleConfig:
    """Minimal config stub so main.py's hasattr(grader.config, ...) checks work."""
    default_workers: int = 2   # 2 students at a time × 3 models each = 6 concurrent calls