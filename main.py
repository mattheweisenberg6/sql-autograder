#!/usr/bin/env python3
"""
Main entry point for SQL Autograder.
Command-line interface for grading submissions and generating statistics.
"""

import argparse
import time
from pathlib import Path

from sqlAutograder import (
    get_gemini_config,
    get_grading_config,
    get_ollama_config,
    get_openai_config,
    GeminiGrader,
    OllamaGrader,
    OpenAIGrader,
    SubmissionLoader,
    ResultsProcessor,
    GradingStatistics,
    GradingVisualizer
)


def get_grader(model: str):
    """
    Get the appropriate grader based on model selection.
    
    Args:
        model: Model identifier ('gemini', OpenAI model name, or Ollama model name)
        
    Returns:
        Tuple of (grader instance, model display name)
    """
    # OpenAI models
    if model in ['gpt-4.1-mini', 'gpt-4.1', 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo']:
        config = get_openai_config(model_name=model)
        grader = OpenAIGrader(config)
        return grader, f"OpenAI ({model})"
    
    # Gemini model
    elif model == 'gemini':
        config = get_gemini_config()
        grader = GeminiGrader(config)
        return grader, f"Gemini ({config.model_name})"
    
    # Ollama models (everything else)
    else:
        config = get_ollama_config(model_name=model)
        grader = OllamaGrader(config)
        return grader, f"Ollama ({model})"


def grade_submissions(
    input_csv: str,
    output_csv: str = None,
    max_students: int = None,
    rate_limit_delay: float = 1.0,
    model: str = 'gemini'
) -> bool:
    """
    Grade student submissions and save results.
    
    Args:
        input_csv: Path to input CSV with submissions
        output_csv: Path to save grading results (default: output/<model>/grading_results.csv)
        max_students: Maximum number of students to grade (None for all)
        rate_limit_delay: Delay between API calls in seconds
        model: Model to use ('gemini', OpenAI model, or Ollama model name)
        
    Returns:
        bool: True if successful
    """
    # Create output directory organized by model name
    if model in ['gpt-4.1-mini', 'gpt-4.1', 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo']:
        model_suffix = f"openai-{model.replace('.', '-')}"
    else:
        model_suffix = model.replace(':', '-').replace('.', '-')
    
    output_dir = Path("output") / model_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_csv is None:
        output_csv = str(output_dir / "grading_results.csv")
    
    print("=" * 60)
    print("SQL AUTOGRADER")
    print("=" * 60)
    print()
    
    # Load configurations
    try:
        print("1. Loading configuration...")
        grading_config = get_grading_config()
        
        # Initialize the appropriate grader based on model selection
        grader, model_display = get_grader(model)
        
        print(f"   ✓ Using model: {model_display}")
        print()
    except ValueError as e:
        print(f"   ✗ Configuration error: {e}")
        return False
    
    # Load submissions
    print("2. Loading submissions...")
    loader = SubmissionLoader(input_csv, grading_config.question_columns)
    
    if not loader.load():
        print("   ✗ Failed to load CSV file")
        return False
    
    print(f"   ✓ Loaded {loader.get_count()} submissions")
    
    # Validate columns
    missing_cols = loader.validate_columns()
    if missing_cols:
        print(f"   ✗ Missing columns: {', '.join(missing_cols)}")
        return False
    
    print("   ✓ All required columns present")
    print()
    
    # Initialize grader
    print(f"3. Initializing {model_display} grader...")
    print("   ✓ Grader initialized")
    if model not in ['gemini', 'gpt-4.1-mini', 'gpt-4.1', 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo']:
        print("   ⚠ Note: Local models may be slower than API calls")
    print()
    
    # Process submissions
    submissions = loader.get_submissions(max_students)
    total = len(submissions)
    
    print(f"4. Grading {total} submissions...")
    print()
    
    results = []
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for i, submission in enumerate(submissions, 1):
        print(f"--- Student {i}/{total} ---")
        print(f"Name: {submission.student_name}")
        print(f"ID: {submission.student_id}")
        
        # Display grader scores
        for q_num in grading_config.questions:
            score = submission.grader_scores[q_num]
            print(f"Q{q_num} Grader Score: {score}/10")
        
        # Grade with LLM
        student_start = time.time()
        llm_result, error = grader.grade_student_submission(submission.queries)
        student_time = time.time() - student_start
        
        if llm_result:
            # Successful grading
            result = ResultsProcessor.create_result_from_grading(
                submission.student_id,
                submission.student_name,
                submission.queries,
                submission.grader_scores,
                llm_result,
                grading_config.questions
            )
            
            # Display LLM scores
            for q_num in grading_config.questions:
                q_prefix = f'q{q_num.replace(".", "_")}'
                llm_score = getattr(result, f'{q_prefix}_llm_score')
                diff = getattr(result, f'{q_prefix}_score_difference')
                print(f"  Q{q_num}: LLM={llm_score:.1f}/10, Diff={diff:+.1f}")
            
            print(f"  Total: LLM={result.total_llm_score:.1f}/50, "
                  f"Grader={result.total_grader_score}/50, "
                  f"Diff={result.total_score_difference:+.1f}")
            print(f"  Time: {student_time:.1f}s")
            
            success_count += 1
        else:
            # Failed grading
            print(f"  ✗ Grading failed: {error}")
            result = ResultsProcessor.create_failed_result(
                submission.student_id,
                submission.student_name,
                submission.queries,
                submission.grader_scores,
                grading_config.questions
            )
            fail_count += 1
        
        results.append(result)
        print()
        
        # Rate limiting
        if i < total:
            time.sleep(rate_limit_delay)
    
    total_time = time.time() - start_time
    
    # Save results
    print("=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    print()
    
    if ResultsProcessor.save_results_to_csv(results, output_csv):
        print(f"✓ Results saved to: {output_csv}")
    else:
        print(f"✗ Failed to save results")
        return False
    
    # Summary
    print()
    print("=" * 60)
    print("GRADING SUMMARY")
    print("=" * 60)
    print(f"Model: {model_display}")
    print(f"Total students: {total}")
    print(f"Successfully graded: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total time: {total_time:.1f}s ({total_time/total:.1f}s per student)")
    print()
    
    return True


def generate_statistics(results_csv: str, output_txt: str = None) -> bool:
    """
    Generate statistics report from grading results.
    
    Args:
        results_csv: Path to grading results CSV
        output_txt: Path to save statistics report (default: same directory as input CSV)
        
    Returns:
        bool: True if successful
    """
    # Use same directory as input CSV with model suffix
    if output_txt is None:
        output_dir = Path(results_csv).parent
        model_suffix = output_dir.name
        output_txt = str(output_dir / f"statistics_report_{model_suffix}.txt")
    
    print("=" * 60)
    print("GENERATING STATISTICS")
    print("=" * 60)
    print()
    
    stats = GradingStatistics(results_csv)
    
    if not stats.load_and_validate():
        print("✗ Failed to load results")
        return False
    
    print(f"✓ Loaded results for analysis")
    print()
    
    if stats.save_report(output_txt):
        print(f"✓ Statistics saved to: {output_txt}")
        
        # Also print to console
        print()
        print(stats.generate_report())
        return True
    else:
        print("✗ Failed to save statistics")
        return False
    
def generate_per_grader_statistics(results_csv: str, output_txt: str = None) -> bool:
    """
    Generate per-grader statistics report from grading results.
    
    Args:
        results_csv: Path to grading results CSV
        output_txt: Path to save statistics report (default: same directory as input CSV)
        
    Returns:
        bool: True if successful
    """
    # Use same directory as input CSV with model suffix
    if output_txt is None:
        output_dir = Path(results_csv).parent
        model_suffix = output_dir.name
        output_txt = str(output_dir / f"per_grader_statistics_{model_suffix}.txt")
    
    print("=" * 60)
    print("GENERATING PER-GRADER STATISTICS")
    print("=" * 60)
    print()
    
    stats = GradingStatistics(results_csv)
    
    if not stats.load_and_validate():
        print("✗ Failed to load results")
        return False
    
    print(f"✓ Loaded results for analysis")
    print()
    
    if stats.save_per_grader_report(output_txt):
        print(f"✓ Per-grader statistics saved to: {output_txt}")
        
        # Also print to console
        print()
        print(stats.generate_per_grader_report())
        return True
    else:
        print("✗ Failed to save per-grader statistics")
        return False
    
def generate_visualizations(results_csv: str, output_dir: str = None) -> bool:
    """
    Generate visualization plots from grading results.
    
    Args:
        results_csv: Path to grading results CSV
        output_dir: Directory to save visualizations (default: same directory as input CSV)
        
    Returns:
        bool: True if successful
    """
    # Use same directory as input CSV if not specified
    if output_dir is None:
        output_dir = str(Path(results_csv).parent)
    
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    print()
    
    visualizer = GradingVisualizer(results_csv)
    
    print("Creating overall distribution plot...")
    visualizer.plot_overall_distribution()
    
    print("\nCreating per-grader distribution plots...")
    visualizer.plot_per_grader_distributions()
    
    print("\nCreating all-graders grid plot...")
    visualizer.plot_all_graders_grid()
    
    print()
    print("=" * 60)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"\nCheck the '{output_dir}/' folder for visualization files.")
    print()
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SQL Autograder - LLM-based SQL grading system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grade all submissions with Gemini (default)
  python main.py grade exam1-submission.csv
  
  # Grade with OpenAI GPT-4.1-mini (recommended)
  python main.py grade exam1-submission.csv --model gpt-4.1-mini
  
  # Grade with OpenAI GPT-4.1
  python main.py grade exam1-submission.csv --model gpt-4.1

  # Grade with OpenAI GPT-4o-mini
  python main.py grade exam1-submission.csv --model gpt-4o-mini
  
  # Grade with OpenAI GPT-4o
  python main.py grade exam1-submission.csv --model gpt-4o
  
  # Grade with DeepSeek-R1 (local Ollama model)
  python main.py grade exam1-submission.csv --model deepseek-r1
  
  # Grade with Llama 3.1 8B (local Ollama model)
  python main.py grade exam1-submission.csv --model llama3.1:8b
  
  # Generate overall statistics
  python main.py stats output/openai-gpt-4-1-mini/grading_results.csv
  
  # Generate per-grader statistics
  python main.py grader-stats output/openai-gpt-4-1-mini/grading_results.csv
  
  # Generate visualizations
  python main.py visualize output/openai-gpt-4-1-mini/grading_results.csv
  
  # Grade first 50 submissions
  python main.py grade exam1-submission.csv --max-students 50 --model gpt-4.1-mini

Output Structure:
  output/
  ├── gemini/
  │   ├── grading_results.csv
  │   ├── statistics_report_gemini.txt
  │   ├── per_grader_statistics_gemini.txt
  │   ├── overall_distribution_gemini.png
  │   ├── G1_distribution_gemini.png
  │   └── ...
  ├── openai-gpt-4-1-mini/
  │   └── ...
  ├── openai-gpt-4o-mini/
  │   └── ...
  ├── llama3-1-8b/
  │   └── ...
  └── deepseek-r1/
      └── ...

Environment Variables:
  GEMINI_API_KEY          Your Google Gemini API key (required for Gemini model)
  OPENAI_API_KEY          Your OpenAI API key (required for OpenAI models)

OpenAI Setup:
  1. Get API key from https://platform.openai.com/api-keys
  2. Set environment variable:
     export OPENAI_API_KEY='your-api-key-here'

Ollama Setup (for local models):
  brew install ollama
  ollama serve
  ollama pull llama3.1:8b

Available Models:
  OpenAI:  gpt-4.1-mini (recommended), gpt-4.1, gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo
  Gemini:  gemini (uses gemini-2.5-flash)
  Ollama:  deepseek-r1, llama3.1:8b, llama3.2:3b, mistral, qwen2.5:7b
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Grade command
    grade_parser = subparsers.add_parser('grade', help='Grade student submissions')
    grade_parser.add_argument('input_csv', help='Input CSV file with submissions')
    grade_parser.add_argument(
        '--output',
        help='Output CSV file for results (default: output/<model>/grading_results.csv)'
    )
    grade_parser.add_argument(
        '--max-students',
        type=int,
        help='Maximum number of students to grade (default: all)'
    )
    grade_parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Delay between API calls in seconds (default: 1.0)'
    )
    grade_parser.add_argument(
        '--model',
        default='gemini',
        help='Model to use: "gemini", OpenAI model (e.g., "gpt-4o-mini"), or Ollama model (e.g., "llama3.1:8b") (default: gemini)'
    )
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Generate overall statistics report')
    stats_parser.add_argument('results_csv', help='Input CSV file with grading results')
    stats_parser.add_argument(
        '--output',
        help='Output text file for statistics (default: same directory as input CSV)'
    )
    
    # Per-grader stats command
    grader_stats_parser = subparsers.add_parser('grader-stats', help='Generate per-grader statistics report')
    grader_stats_parser.add_argument('results_csv', help='Input CSV file with grading results')
    grader_stats_parser.add_argument(
        '--output',
        help='Output text file for statistics (default: same directory as input CSV)'
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate visualization plots')
    viz_parser.add_argument('results_csv', help='Input CSV file with grading results')
    viz_parser.add_argument(
        '--output-dir',
        help='Output directory for plots (default: same directory as input CSV)'
    )
    args = parser.parse_args()
    
    if args.command == 'grade':
        success = grade_submissions(
            args.input_csv,
            args.output,
            args.max_students,
            args.rate_limit,
            args.model
        )
        exit(0 if success else 1)
    
    elif args.command == 'stats':
        success = generate_statistics(args.results_csv, args.output)
        exit(0 if success else 1)
    
    elif args.command == 'grader-stats':
        success = generate_per_grader_statistics(args.results_csv, args.output)
        exit(0 if success else 1)

    elif args.command == 'visualize':
        success = generate_visualizations(args.results_csv, args.output_dir)
        exit(0 if success else 1)
    
    else:
        parser.print_help()
        exit(1)


if __name__ == '__main__':
    main()