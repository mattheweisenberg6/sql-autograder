"""
Results processing and storage module.
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .calibration import ScoreCalibrator


@dataclass
class GradingResult:
    """Represents the grading result for a single student."""
    student_id: str
    student_name: str

    # Per-question results
    q4_1_query: str
    q4_1_grader_score: float
    q4_1_llm_score: float
    q4_1_raw_llm_score: float        # pre-calibration score
    q4_1_score_difference: float
    q4_1_feedback: str
    q4_1_needs_review: bool

    q4_2_query: str
    q4_2_grader_score: float
    q4_2_llm_score: float
    q4_2_raw_llm_score: float
    q4_2_score_difference: float
    q4_2_feedback: str
    q4_2_needs_review: bool

    q4_3_query: str
    q4_3_grader_score: float
    q4_3_llm_score: float
    q4_3_raw_llm_score: float
    q4_3_score_difference: float
    q4_3_feedback: str
    q4_3_needs_review: bool

    q4_4_query: str
    q4_4_grader_score: float
    q4_4_llm_score: float
    q4_4_raw_llm_score: float
    q4_4_score_difference: float
    q4_4_feedback: str
    q4_4_needs_review: bool

    q4_5_query: str
    q4_5_grader_score: float
    q4_5_llm_score: float
    q4_5_raw_llm_score: float
    q4_5_score_difference: float
    q4_5_feedback: str
    q4_5_needs_review: bool

    # Totals
    total_llm_score: float
    total_raw_llm_score: float       # pre-calibration total
    total_grader_score: float
    total_score_difference: float


class ResultsProcessor:
    """Processes and stores grading results."""

    def __init__(self, calibrator: Optional[ScoreCalibrator] = None):
        """
        Initialize the results processor.

        Args:
            calibrator: Optional ScoreCalibrator instance. If None, a default
                        'curve' calibrator is used. Pass ScoreCalibrator(mode='none')
                        to disable calibration entirely.
        """
        self.calibrator = calibrator if calibrator is not None else ScoreCalibrator(mode="curve")

    @staticmethod
    def create_result_from_grading(
        student_id: str,
        student_name: str,
        queries: Dict[str, str],
        grader_scores: Dict[str, float],
        llm_result: Dict,
        questions: List[str],
        calibrator: Optional[ScoreCalibrator] = None,
    ) -> 'GradingResult':
        """
        Create a GradingResult object from raw grading data.

        Args:
            student_id: Student identifier
            student_name: Student name
            queries: Dictionary of SQL queries by question
            grader_scores: Dictionary of human grader scores
            llm_result: Dictionary with LLM grading results
            questions: List of question numbers
            calibrator: Optional ScoreCalibrator. Defaults to curve-mode calibration.

        Returns:
            GradingResult object with calibrated llm_score and raw_llm_score fields.
        """
        if calibrator is None:
            calibrator = ScoreCalibrator(mode="curve")

        # Apply calibration to entire result dict
        calibrated_result = calibrator.calibrate_result(llm_result)

        total_llm_score = 0.0
        total_raw_llm_score = 0.0
        total_grader_score = 0.0

        result_dict = {
            'student_id': student_id,
            'student_name': student_name,
        }

        for q_num in questions:
            # Support both key formats: question_4_1 and question_4.1
            q_key_underscore = f'question_{q_num.replace(".", "_")}'
            q_key_dot = f'question_{q_num}'
            q_result = calibrated_result.get(q_key_underscore) or calibrated_result.get(q_key_dot) or {}

            llm_score = int(round(q_result.get('score', 0.0)))
            raw_llm_score = int(round(q_result.get('raw_score', llm_score)))   # fallback if no calibration
            grader_score = grader_scores.get(q_num, 0.0)
            score_diff = int(round(llm_score - grader_score))

            total_llm_score += llm_score
            total_raw_llm_score += raw_llm_score
            total_grader_score += grader_score

            q_prefix = f'q{q_num.replace(".", "_")}'
            result_dict.update({
                f'{q_prefix}_query': queries.get(q_num, ''),
                f'{q_prefix}_grader_score': grader_score,
                f'{q_prefix}_llm_score': llm_score,
                f'{q_prefix}_raw_llm_score': raw_llm_score,
                f'{q_prefix}_score_difference': score_diff,
                f'{q_prefix}_feedback': q_result.get('feedback', ''),
                f'{q_prefix}_needs_review': q_result.get('needs_review', False),
            })

        result_dict.update({
            'total_llm_score': int(round(total_llm_score)),
            'total_raw_llm_score': int(round(total_raw_llm_score)),
            'total_grader_score': total_grader_score,
            'total_score_difference': int(round(total_llm_score - total_grader_score)),
        })

        return GradingResult(**result_dict)

    @staticmethod
    def create_failed_result(
        student_id: str,
        student_name: str,
        queries: Dict[str, str],
        grader_scores: Dict[str, float],
        questions: List[str],
    ) -> 'GradingResult':
        """
        Create a GradingResult for a failed grading attempt.

        Args:
            student_id: Student identifier
            student_name: Student name
            queries: Dictionary of SQL queries
            grader_scores: Dictionary of human grader scores
            questions: List of question numbers

        Returns:
            GradingResult with -1 scores indicating failure
        """
        result_dict = {
            'student_id': student_id,
            'student_name': student_name,
        }

        for q_num in questions:
            q_prefix = f'q{q_num.replace(".", "_")}'
            result_dict.update({
                f'{q_prefix}_query': queries.get(q_num, ''),
                f'{q_prefix}_grader_score': grader_scores.get(q_num, 0.0),
                f'{q_prefix}_llm_score': -1.0,
                f'{q_prefix}_raw_llm_score': -1.0,
                f'{q_prefix}_score_difference': -1.0,
                f'{q_prefix}_feedback': 'Grading failed - requires manual review',
                f'{q_prefix}_needs_review': True,
            })

        result_dict.update({
            'total_llm_score': -1.0,
            'total_raw_llm_score': -1.0,
            'total_grader_score': sum(grader_scores.values()),
            'total_score_difference': -1.0,
        })

        return GradingResult(**result_dict)

    @staticmethod
    def save_results_to_csv(results: List['GradingResult'], output_path: str) -> bool:
        """
        Save grading results to a CSV file.

        Args:
            results: List of GradingResult objects
            output_path: Path to save the CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            results_dicts = [asdict(result) for result in results]
            df = pd.DataFrame(results_dicts)
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False