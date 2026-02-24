"""
LLM-based grading module using OpenAI API.
Supports GPT-4.1-mini, GPT-4.1, GPT-4o-mini, GPT-4o, GPT-4-turbo, and GPT-3.5-turbo models.
"""

import json
import time
from typing import Dict, Optional, Tuple
from openai import OpenAI

from .config import OpenAIConfig
from .prompts import create_grading_prompt


SUPPORTED_MODELS = [
    "gpt-4.1-mini",   # Recommended: fast, cost-effective, high quality
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]


class OpenAIGrader:
    """Handles LLM-based grading using OpenAI API.
    
    Supported models:
        - gpt-4.1-mini  (recommended: fast, cost-effective, high quality)
        - gpt-4.1
        - gpt-4o-mini
        - gpt-4o
        - gpt-4-turbo
        - gpt-3.5-turbo
    """
    
    def __init__(self, config: OpenAIConfig):
        """
        Initialize the OpenAI grader.
        
        Args:
            config: OpenAI API configuration
        """
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
    
    def grade_student_submission(
        self, 
        student_queries: Dict[str, str]
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Grade a student's submission for all questions.
        
        Args:
            student_queries: Dictionary mapping question numbers to SQL queries
            
        Returns:
            Tuple of (grading_result, error_message)
            - grading_result: Dictionary with scores and feedback if successful
            - error_message: Error description if grading failed
        """
        prompt = create_grading_prompt(student_queries)
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )
                
                response_text = response.choices[0].message.content
                result = self._parse_response(response_text)
                return result, None
                
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing error on attempt {attempt + 1}: {str(e)}"
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
    
    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse and clean the LLM response.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Dictionary with parsed grading results
            
        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        # Clean markdown code blocks if present
        cleaned_text = response_text.strip()
        
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
            
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        # Try to find JSON object in the response
        # Look for the outermost { } pair
        first_brace = cleaned_text.find('{')
        last_brace = cleaned_text.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned_text = cleaned_text[first_brace:last_brace + 1]
        
        return json.loads(cleaned_text)
    
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