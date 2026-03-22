"""
LLM-based grading module using Google Gemini API.

v1.4: Migrated from deprecated google.generativeai to google.genai SDK.
v1.3 performance optimisations:
  - Uses create_grading_prompt_full() — leaner combined prompt (~1,250 tokens, was ~3,800)
  - max_output_tokens capped at 600
  - retry_delay reduced from 2.0s → 0.5s
  - grade_batch() runs N students concurrently via ThreadPoolExecutor
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from google import genai
from google.genai import types

from .config import GeminiConfig
from .prompts import create_grading_prompt_full


class GeminiGrader:
    """Handles LLM-based grading using Google Gemini API."""

    def __init__(self, config: GeminiConfig):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)

    def grade_student_submission(
        self,
        student_queries: Dict[str, str],
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Grade a single student's submission.

        Returns:
            Tuple of (grading_result, error_message)
        """
        # Use full prompt (context + student queries) — Gemini has no separate system turn
        prompt = create_grading_prompt_full(student_queries)

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=1500,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                return self._parse_response(response.text), None

            except json.JSONDecodeError as e:
                print(f"\n[DEBUG] Attempt {attempt + 1} — raw response ({len(response.text)} chars):")
                print(repr(response.text[:500]))
                print()
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
            max_workers: Max concurrent API calls (default 5)

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