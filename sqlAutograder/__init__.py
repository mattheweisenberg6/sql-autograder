"""
SQL Autograder - LLM-based SQL grading system using Google Gemini API, OpenAI API, and Ollama.
"""

from .config import get_gemini_config, get_grading_config, get_ollama_config, get_openai_config, get_claude_config
from .grader import GeminiGrader
from .ollama_grader import OllamaGrader
from .openai_grader import OpenAIGrader
from .claude_grader import ClaudeGrader
from .data_loader import SubmissionLoader
from .results import ResultsProcessor
from .statistics import GradingStatistics
from .visualizations import GradingVisualizer
from .calibration import ScoreCalibrator

__version__ = "1.3.0"

__all__ = [
    'get_gemini_config',
    'get_grading_config',
    'get_ollama_config',
    'get_openai_config',
    'get_claude_config',
    'GeminiGrader',
    'OllamaGrader',
    'OpenAIGrader',
    'ClaudeGrader',
    'SubmissionLoader',
    'ResultsProcessor',
    'GradingStatistics',
    'GradingVisualizer',
    'ScoreCalibrator',
]