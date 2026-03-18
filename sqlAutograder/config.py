"""
Configuration management for SQL Autograder.
Handles API keys, model settings, and grading parameters.
"""

import os
from dataclasses import dataclass, field


@dataclass
class GeminiConfig:
    """Configuration for Gemini API."""
    api_key: str
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class OllamaConfig:
    """Configuration for Ollama local models."""
    model_name: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 8192
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: float = 300.0


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API."""
    api_key: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 600        # grading responses are ~250 tokens
    max_retries: int = 3
    retry_delay: float = 0.5     # reduced from 2.0s
    timeout: float = 60.0        # reasoning models (o4-mini, o3-mini) need more time
    is_reasoning_model: bool = False  # True for o-series: skips temperature, uses max_completion_tokens

# Models that use reasoning (o-series) — temperature not supported, slower responses
_REASONING_MODELS = frozenset({"o1", "o1-mini", "o3", "o3-mini", "o4-mini"})


@dataclass
class GradingConfig:
    """Configuration for grading parameters."""
    questions: list[str] = field(default_factory=lambda: ['4.1', '4.2', '4.3', '4.4', '4.5'])
    points_per_question: int = 10
    total_points: int = 50
    question_columns: dict[str, dict[str, str]] = field(default_factory=dict)


def get_gemini_config() -> GeminiConfig:
    """
    Get Gemini API configuration from environment variables.
    
    Returns:
        GeminiConfig: Configuration object with API settings
        
    Raises:
        ValueError: If GEMINI_API_KEY environment variable is not set
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it using: export GEMINI_API_KEY='your-api-key'"
        )
    
    return GeminiConfig(api_key=api_key)


def get_ollama_config(model_name: str = "llama3.1:8b") -> OllamaConfig:
    """
    Get Ollama configuration for local models.
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        OllamaConfig: Configuration object with Ollama settings
    """
    return OllamaConfig(model_name=model_name)


def get_openai_config(model_name: str = "gpt-4o-mini") -> OpenAIConfig:
    """
    Get OpenAI API configuration from environment variables.

    Args:
        model_name: Model to use. Standard: gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo
                    Reasoning: o4-mini (recommended), o3-mini, o3, o1-mini, o1

    Returns:
        OpenAIConfig: Configuration object with API settings

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it using: export OPENAI_API_KEY='your-api-key'"
        )

    is_reasoning = model_name in _REASONING_MODELS
    return OpenAIConfig(
        api_key=api_key,
        model_name=model_name,
        is_reasoning_model=is_reasoning,
        # Reasoning models are slower — give them more time
        timeout=120.0 if is_reasoning else 60.0,
    )


def get_grading_config() -> GradingConfig:
    """
    Get grading configuration.
    
    Returns:
        GradingConfig: Configuration object with grading parameters
    
    NOTE: The column names below must match your CSV file exactly.
    If your CSV has different column names, update them here.
    
    Expected CSV columns:
    - 'Question 4.1 Response' (SQL query)
    - 'Question 4.1 Score' (human grader score)
    - 'Question 4.2 Response' (SQL query)
    - 'Question 4.2 Score' (human grader score)
    ... and so on for questions 4.3, 4.4, 4.5
    """
    questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
    
    # IMPORTANT: Update these column names if your CSV uses different names
    question_columns = {
        '4.1': {'response': 'Question 4.1 Response', 'score': 'Question 4.1 Score'},
        '4.2': {'response': 'Question 4.2 Response', 'score': 'Question 4.2 Score'},
        '4.3': {'response': 'Question 4.3 Response', 'score': 'Question 4.3 Score'},
        '4.4': {'response': 'Question 4.4 Response', 'score': 'Question 4.4 Score'},
        '4.5': {'response': 'Question 4.5 Response', 'score': 'Question 4.5 Score'}
    }
    
    return GradingConfig(
        questions=questions,
        points_per_question=10,
        total_points=50,
        question_columns=question_columns
    )