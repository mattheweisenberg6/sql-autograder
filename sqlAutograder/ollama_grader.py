"""
LLM-based grading module using Ollama for local models.
Supports DeepSeek-R1, Llama, and other Ollama-compatible models.
"""

import json
import re
import time
import requests
from typing import Dict, List, Optional, Tuple

from .config import OllamaConfig
from .prompts import create_grading_prompt, create_single_question_prompt


class OllamaGrader:
    """Handles LLM-based grading using local Ollama models."""
    
    # Models known to only output one question at a time
    PER_QUESTION_MODELS = {'mistral', 'mistral:7b', 'mistral:latest'}

    def __init__(self, config: OllamaConfig, per_question_mode: bool = None):
        """
        Initialize the Ollama grader.
        
        Args:
            config: Ollama configuration
            per_question_mode: If True, grade one question per API call and merge results.
                               Defaults to True for known models (mistral), False otherwise.
        """
        self.config = config
        self.api_url = f"{config.base_url}/api/generate"
        if per_question_mode is None:
            self.per_question_mode = config.model_name in self.PER_QUESTION_MODELS
        else:
            self.per_question_mode = per_question_mode
    
    def _check_server(self) -> bool:
        """
        Check if Ollama server is running.
        
        Returns:
            bool: True if server is accessible
        """
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def _strip_thinking(self, text: str) -> str:
        """
        Remove <think>...</think> tags from R1 model output.
        
        Args:
            text: Raw response text
            
        Returns:
            Text with thinking tags removed
        """
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned.strip()

    def _extract_all_json_blocks(self, text: str) -> List[Dict]:
        """
        Extract all top-level JSON objects from text.
        Handles models that return one JSON block per question instead of
        one combined object (e.g. llama3.1:8b).

        Args:
            text: Text potentially containing multiple JSON objects

        Returns:
            List of parsed JSON dicts found in the text
        """
        objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                objects.append(json.loads(text[start:j + 1]))
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
            i += 1
        return objects

    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse and clean the LLM response.
        Handles both a single combined JSON object and multiple per-question
        JSON objects (as some models like llama3.1:8b return one block per question).

        Args:
            response_text: Raw response text from LLM

        Returns:
            Dictionary with parsed grading results

        Raises:
            json.JSONDecodeError: If response cannot be parsed into a valid result
        """
        # Strip thinking tags (for R1 models)
        cleaned_text = self._strip_thinking(response_text)

        # Strip markdown code fences
        if "```json" in cleaned_text:
            start = cleaned_text.find("```json") + 7
            end = cleaned_text.find("```", start)
            if end != -1:
                cleaned_text = cleaned_text[start:end]
        elif "```" in cleaned_text:
            start = cleaned_text.find("```") + 3
            end = cleaned_text.find("```", start)
            if end != -1:
                cleaned_text = cleaned_text[start:end]

        cleaned_text = cleaned_text.strip()

        # --- Strategy 1: try parsing the outermost { } as one combined object ---
        first_brace = cleaned_text.find('{')
        last_brace = cleaned_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = cleaned_text[first_brace:last_brace + 1]
            try:
                result = json.loads(candidate)
                if any(k.startswith('question_4') for k in result):
                    return result
            except json.JSONDecodeError:
                pass

        # --- Strategy 2: model returned one JSON block per question — merge them ---
        blocks = self._extract_all_json_blocks(cleaned_text)
        if blocks:
            merged = {}
            for block in blocks:
                for key, value in block.items():
                    if key.startswith('question_4'):
                        merged[key] = value
            if merged:
                return merged

        raise json.JSONDecodeError("Could not parse grading response", cleaned_text, 0)

    def _grade_per_question(
        self,
        student_queries: Dict[str, str]
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Grade each question with a separate API call and merge results.
        Used for models that truncate or ignore questions when given all 5 at once.

        Args:
            student_queries: Dictionary mapping question numbers to SQL queries

        Returns:
            Tuple of (merged_grading_result, error_message)
        """
        questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
        merged = {}

        for q_num in questions:
            prompt = create_single_question_prompt(q_num, student_queries.get(q_num, '[NO ANSWER PROVIDED]'))

            for attempt in range(self.config.max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        json={
                            "model": self.config.model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": self.config.temperature,
                                "num_predict": self.config.max_tokens
                            }
                        },
                        timeout=self.config.timeout
                    )

                    if response.status_code != 200:
                        error_msg = f"Ollama API error on Q{q_num}: {response.status_code}"
                        if attempt < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay)
                            continue
                        return None, error_msg

                    result_json = response.json()
                    response_text = result_json.get("response", "")

                    if not response_text:
                        if attempt < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay)
                            continue
                        return None, f"Empty response for Q{q_num}"

                    parsed = self._parse_response(response_text)
                    merged.update(parsed)
                    break  # success for this question

                except json.JSONDecodeError as e:
                    error_msg = f"JSON parse error on Q{q_num} attempt {attempt + 1}: {e}"
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                    else:
                        return None, error_msg
                except requests.exceptions.Timeout:
                    error_msg = f"Timeout on Q{q_num} attempt {attempt + 1}"
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                    else:
                        return None, error_msg
                except Exception as e:
                    error_msg = f"Error on Q{q_num} attempt {attempt + 1}: {e}"
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                    else:
                        return None, error_msg

        return merged, None

    def grade_student_submission(
        self, 
        student_queries: Dict[str, str]
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Grade a student's submission for all questions.
        Automatically uses per-question mode for models known to truncate multi-question responses.
        
        Args:
            student_queries: Dictionary mapping question numbers to SQL queries
            
        Returns:
            Tuple of (grading_result, error_message)
            - grading_result: Dictionary with scores and feedback if successful
            - error_message: Error description if grading failed
        """
        # Check if server is running
        if not self._check_server():
            return None, (
                "Ollama server not running. Start it with: ollama serve\n"
                "Then pull the model with: ollama pull " + self.config.model_name
            )

        if self.per_question_mode:
            return self._grade_per_question(student_queries)

        prompt = create_grading_prompt(student_queries)
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.config.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens
                        }
                    },
                    timeout=self.config.timeout
                )
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                        continue
                    return None, error_msg
                
                result_json = response.json()
                response_text = result_json.get("response", "")
                
                if not response_text:
                    error_msg = "Empty response from Ollama"
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                        continue
                    return None, error_msg
                
                result = self._parse_response(response_text)
                return result, None
                
            except json.JSONDecodeError as e:
                print(f"\n  [DEBUG] Raw response length: {len(response_text)} chars (attempt {attempt + 1})")
                print(f"  {response_text[:2000]}...")
                print()
                error_msg = f"JSON parsing error on attempt {attempt + 1}: {str(e)}"
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return None, error_msg
                    
            except requests.exceptions.Timeout:
                error_msg = f"Request timeout on attempt {attempt + 1}"
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return None, error_msg
                    
            except Exception as e:
                error_msg = f"Grading error on attempt {attempt + 1}: {str(e)}"
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return None, error_msg
        
        return None, "Max retries exceeded"
    
    @staticmethod
    def create_failed_result(
        question_numbers: list[str],
        error_message: str = "Grading failed"
    ) -> Dict:
        """
        Create a result dictionary for failed grading attempts.
        
        Args:
            question_numbers: List of question numbers
            error_message: Error description
            
        Returns:
            Dictionary with -1 scores and error information
        """
        result = {}
        
        for q_num in question_numbers:
            q_key = f'question_{q_num.replace(".", "_")}'
            result[q_key] = {
                'score': -1,
                'deduction_details': error_message,
                'feedback': 'Automatic grading failed - requires manual review',
                'needs_review': True
            }
        
        return result