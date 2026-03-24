"""
LLM-based grading module using Anthropic Claude API.
Supports claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5-20251001, and other Claude models.

Mirrors the OpenAI grader pattern:
  - System message carries the grading context once per call
  - grade_batch() runs N students concurrently via ThreadPoolExecutor
  - max_tokens capped at 1500 (grading response is ~600-900 tokens)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import anthropic

from .config import ClaudeConfig
from .prompts import create_grading_prompt, create_system_prompt

# Backoff delays (seconds) for successive 429 rate-limit retries
_RATE_LIMIT_BACKOFF = [15, 30, 60]


SUPPORTED_MODELS = [
    "claude-sonnet-4-6",           # recommended — best balance of speed, cost, accuracy
    "claude-opus-4-6",             # highest accuracy, slower and more expensive
    "claude-haiku-4-5-20251001",   # fastest and cheapest, good for large batches
]


class ClaudeGrader:
    """Handles LLM-based grading using Anthropic Claude API."""

    def __init__(self, config: ClaudeConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self._system_prompt = create_system_prompt()

    def grade_student_submission(
        self,
        student_queries: Dict[str, str],
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Grade a single student's submission.
        Handles 429 rate-limit errors with exponential backoff separately from
        other errors so retries aren't wasted on non-recoverable failures.

        Returns:
            Tuple of (grading_result, error_message)
        """
        prompt = create_grading_prompt(student_queries)
        rate_limit_attempt = 0
        attempt = 0

        while attempt < self.config.max_retries:
            try:
                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    system=self._system_prompt,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                response_text = response.content[0].text
                if not response_text or not response_text.strip():
                    raise json.JSONDecodeError("Empty response from model", "", 0)
                return self._parse_response(response_text), None

            except json.JSONDecodeError as e:
                raw_preview = repr(response_text[:200]) if 'response_text' in locals() else "None"
                print(f"  [DEBUG] Attempt {attempt + 1}: unparseable response. Preview: {raw_preview}")
                error_msg = f"JSON parsing error on attempt {attempt + 1}: {e}"
                attempt += 1
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                else:
                    return None, error_msg

            except anthropic.RateLimitError:
                # 429 — back off without burning a normal retry attempt
                wait = _RATE_LIMIT_BACKOFF[min(rate_limit_attempt, len(_RATE_LIMIT_BACKOFF) - 1)]
                rate_limit_attempt += 1
                print(f"  [RATE LIMIT] Waiting {wait}s before retry "
                      f"(rate limit hit #{rate_limit_attempt})...")
                time.sleep(wait)
                if rate_limit_attempt > len(_RATE_LIMIT_BACKOFF):
                    return None, f"Rate limit exceeded after {rate_limit_attempt} backoffs"
                # Do NOT increment attempt — retry the same slot

            except Exception as e:
                error_msg = f"Grading error on attempt {attempt + 1}: {e}"
                attempt += 1
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
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

        # Extract outermost { } in case of any leading/trailing prose
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