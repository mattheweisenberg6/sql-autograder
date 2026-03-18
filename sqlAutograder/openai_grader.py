"""
LLM-based grading module using OpenAI API.
Supports GPT-4.1-mini, GPT-4.1, GPT-4o-mini, GPT-4o, GPT-4-turbo, and GPT-3.5-turbo.

v1.3 performance optimisations:
  - System message carries CALIBRATION_CONTEXT once per session (not per student)
  - max_tokens reduced from 4096 → 600 (response is ~250 tokens)
  - retry_delay reduced from 2.0s → 0.5s
  - grade_batch() runs N students concurrently via ThreadPoolExecutor
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

from .config import OpenAIConfig
from .prompts import create_grading_prompt, create_system_prompt


SUPPORTED_MODELS = [
    # Reasoning models (recommended for grading accuracy)
    "o4-mini",      # best balance of speed, cost, accuracy
    "o3-mini",
    "o3",
    "o1-mini",
    "o1",
    # Standard models
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]


class OpenAIGrader:
    """Handles LLM-based grading using OpenAI API."""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        # System message sent once per API call — not repeated per student in batch mode
        self._system_message = {"role": "system", "content": create_system_prompt()}

    def grade_student_submission(
        self,
        student_queries: Dict[str, str],
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Grade a single student's submission.

        Returns:
            Tuple of (grading_result, error_message)
        """
        prompt = create_grading_prompt(student_queries)

        for attempt in range(self.config.max_retries):
            try:
                # Reasoning models (o-series) do not support temperature or max_tokens;
                # they use max_completion_tokens instead and think internally.
                if self.config.is_reasoning_model:
                    api_kwargs = dict(
                        model=self.config.model_name,
                        messages=[
                            self._system_message,
                            {"role": "user", "content": prompt},
                        ],
                        max_completion_tokens=5000,  # reasoning models spend tokens thinking before outputting
                        timeout=self.config.timeout,
                    )
                else:
                    api_kwargs = dict(
                        model=self.config.model_name,
                        messages=[
                            self._system_message,
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.config.temperature,
                        max_tokens=600,
                        timeout=self.config.timeout,
                    )
                response = self.client.chat.completions.create(**api_kwargs)
                response_text = response.choices[0].message.content
                if not response_text or not response_text.strip():
                    raise json.JSONDecodeError("Empty response from model", "", 0)
                return self._parse_response(response_text), None

            except json.JSONDecodeError as e:
                raw_preview = repr(response_text[:200]) if response_text else "None"
                print(f"  [DEBUG] Attempt {attempt + 1}: empty/unparseable response. Preview: {raw_preview}")
                error_msg = f"JSON parsing error on attempt {attempt + 1}: {e}"
                if attempt < self.config.max_retries - 1:
                    time.sleep(0.5)
                else:
                    return None, error_msg

            except Exception as e:
                error_msg = f"Grading error on attempt {attempt + 1}: {e}"
                if attempt < self.config.max_retries - 1:
                    time.sleep(0.5)
                else:
                    return None, error_msg

        return None, "Max retries exceeded"

    def grade_batch(
        self,
        submissions: List[Tuple[str, Dict[str, str]]],
        max_workers: int = 5,
    ) -> Dict[str, Tuple[Optional[Dict], Optional[str]]]:
        """
        Grade multiple students concurrently.

        Args:
            submissions: List of (student_id, queries_dict) tuples
            max_workers: Max concurrent API calls (default 5 — safe for tier-1 rate limits)

        Returns:
            Dict mapping student_id → (grading_result, error_message)
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

    def _parse_response(self, response_text: str) -> Dict:
        """Parse and clean the LLM response."""
        cleaned = response_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]

        return json.loads(cleaned)

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