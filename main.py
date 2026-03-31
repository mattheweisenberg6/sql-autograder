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
    get_claude_config,
    GeminiGrader,
    OllamaGrader,
    OpenAIGrader,
    ClaudeGrader,
    EnsembleGrader,
    SubmissionLoader,
    ResultsProcessor,
    GradingStatistics,
    GradingVisualizer,
    ScoreCalibrator,
)


_OPENAI_MODELS = frozenset([
    'o4-mini', 'o3-mini', 'o3', 'o1-mini', 'o1',
    'gpt-4.1-mini', 'gpt-4.1', 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo',
])
_CLAUDE_MODELS = frozenset([
    'claude-sonnet-4-6', 'claude-opus-4-6', 'claude-haiku-4-5-20251001',
    # legacy aliases for convenience
    'claude-sonnet', 'claude-opus', 'claude-haiku',
])
_CLAUDE_ALIASES = {
    'claude-sonnet': 'claude-sonnet-4-6',
    'claude-opus':   'claude-opus-4-6',
    'claude-haiku':  'claude-haiku-4-5-20251001',
}


def get_grader(model: str):
    """
    Get the appropriate grader based on model selection.

    Args:
        model: Model identifier ('gemini', Claude model name, OpenAI model name, or Ollama model name)

    Returns:
        Tuple of (grader instance, model display name)
    """
    # Resolve short aliases (claude-sonnet → claude-sonnet-4-6, etc.)
    model = _CLAUDE_ALIASES.get(model, model)

    # Ensemble model
    if model == 'ensemble':
        grader = EnsembleGrader()
        return grader, "Ensemble (Gemini + o4-mini + Claude, median vote)"

    # Claude models
    if model in _CLAUDE_MODELS:
        config = get_claude_config(model_name=model)
        grader = ClaudeGrader(config)
        return grader, f"Claude ({model})"

    # OpenAI models
    if model in _OPENAI_MODELS:
        config = get_openai_config(model_name=model)
        grader = OpenAIGrader(config)
        return grader, f"OpenAI ({model})"

    # Gemini model
    if model == 'gemini':
        config = get_gemini_config()
        grader = GeminiGrader(config)
        return grader, f"Gemini ({config.model_name})"

    # Ollama models (everything else)
    config = get_ollama_config(model_name=model)
    grader = OllamaGrader(config)
    return grader, f"Ollama ({model})"


def grade_submissions(
    input_csv: str,
    output_csv: str = None,
    max_students: int = None,
    rate_limit_delay: float = 0.0,
    model: str = 'gemini',
    calibration: bool = True,
    concurrent_workers: int = 5,
) -> bool:
    """
    Grade student submissions and save results.

    Args:
        input_csv: Path to input CSV with submissions
        output_csv: Path to save grading results (default: output/<model>/grading_results.csv)
        max_students: Maximum number of students to grade (None for all)
        rate_limit_delay: Delay between batches in seconds (default 0 — not needed with low workers)
        model: Model to use ('gemini', OpenAI model, or Ollama model name)
        calibration: Apply post-processing grade curve to correct LLM strictness bias
        concurrent_workers: Number of parallel API calls (default 5, use 1 to disable)

    Returns:
        bool: True if successful
    """
    # Create output directory organized by model name
    model = _CLAUDE_ALIASES.get(model, model)  # resolve aliases before suffix
    if model == 'ensemble':
        model_suffix = 'ensemble'
    elif model in _OPENAI_MODELS:
        model_suffix = f"openai-{model.replace('.', '-')}"
    elif model in _CLAUDE_MODELS:
        model_suffix = f"claude-{model.replace('.', '-').replace(':', '-')}"
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

        # Use the grader's recommended worker count unless the user explicitly passed --workers
        if concurrent_workers == 5 and hasattr(grader.config, 'default_workers'):
            concurrent_workers = grader.config.default_workers

        print(f"   ✓ Using model: {model_display}")
        
        # Initialize calibrator
        calibrator = ScoreCalibrator(mode="curve") if calibration else ScoreCalibrator(mode="none")
        if calibration:
            print(f"   ✓ Grade calibration: ENABLED (curve mode, 80% bias correction)")
        else:
            print(f"   ⚠ Grade calibration: DISABLED (raw LLM scores will be used)")
        print(f"   ✓ Concurrent workers: {concurrent_workers}")
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
    if model not in _OPENAI_MODELS and model not in _CLAUDE_MODELS and model != 'gemini':
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

    # Build a lookup so we can access submission details by student_id
    submission_map = {sub.student_id: sub for sub in submissions}
    completed_count = 0

    def _print_student_result(submission, llm_result, error, elapsed):
        """Print a single student's grading result immediately."""
        nonlocal success_count, fail_count, completed_count
        completed_count += 1

        print(f"--- Student {completed_count}/{total} ---")
        print(f"Name: {submission.student_name}")
        print(f"ID: {submission.student_id}")

        for q_num in grading_config.questions:
            score = submission.grader_scores[q_num]
            print(f"Q{q_num} Grader Score: {score}/10")

        if llm_result:
            result = ResultsProcessor.create_result_from_grading(
                submission.student_id,
                submission.student_name,
                submission.queries,
                submission.grader_scores,
                llm_result,
                grading_config.questions,
                calibrator=calibrator,
            )

            for q_num in grading_config.questions:
                q_prefix = f'q{q_num.replace(".", "_")}'
                llm_score = getattr(result, f'{q_prefix}_llm_score')
                raw_score = getattr(result, f'{q_prefix}_raw_llm_score')
                diff = getattr(result, f'{q_prefix}_score_difference')
                if calibration and raw_score != llm_score:
                    print(f"  Q{q_num}: LLM={llm_score}/10 (raw={raw_score}), Diff={diff:+d}")
                else:
                    print(f"  Q{q_num}: LLM={llm_score}/10, Diff={diff:+d}")

            raw_total = result.total_raw_llm_score
            cal_total = result.total_llm_score
            if calibration and raw_total != cal_total:
                print(f"  Total: LLM={cal_total}/50 (raw={raw_total}), "
                      f"Grader={result.total_grader_score}/50, "
                      f"Diff={result.total_score_difference:+d}")
            else:
                print(f"  Total: LLM={cal_total}/50, "
                      f"Grader={result.total_grader_score}/50, "
                      f"Diff={result.total_score_difference:+d}")
            print(f"  Time: {elapsed:.1f}s")
            success_count += 1
        else:
            print(f"  ✗ Grading failed: {error}")
            result = ResultsProcessor.create_failed_result(
                submission.student_id,
                submission.student_name,
                submission.queries,
                submission.grader_scores,
                grading_config.questions,
            )
            fail_count += 1

        print()
        return result

    if concurrent_workers > 1:
        # Concurrent mode: print each student as soon as their API call completes
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"   Running {concurrent_workers} concurrent API calls — results print as each completes.")
        print()

        # results_map preserves final ordering for CSV output
        results_map = {}

        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            future_to_sub = {
                executor.submit(grader.grade_student_submission, sub.queries): sub
                for sub in submissions
            }
            for future in as_completed(future_to_sub):
                sub = future_to_sub[future]
                t_start = time.time()
                try:
                    llm_result, error = future.result()
                except Exception as e:
                    llm_result, error = None, str(e)
                elapsed = time.time() - t_start

                result = _print_student_result(sub, llm_result, error, elapsed)
                results_map[sub.student_id] = result

        # Restore original submission order for CSV
        results = [results_map[sub.student_id] for sub in submissions]

    else:
        # Sequential mode: grade and print one at a time
        print()
        for submission in submissions:
            t_start = time.time()
            llm_result, error = grader.grade_student_submission(submission.queries)
            elapsed = time.time() - t_start

            result = _print_student_result(submission, llm_result, error, elapsed)
            results.append(result)

            if rate_limit_delay > 0:
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
    print(f"Calibration: {'ENABLED (curve mode)' if calibration else 'DISABLED'}")
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

  # Grade with Claude Sonnet (recommended — best balance)
  python main.py grade exam1-submission.csv --model claude-sonnet

  # Grade with Claude Sonnet (full model name)
  python main.py grade exam1-submission.csv --model claude-sonnet-4-6

  # Grade with Claude Opus (highest accuracy)
  python main.py grade exam1-submission.csv --model claude-opus

  # Grade with Claude Haiku (fastest, cheapest — good for large batches)
  python main.py grade exam1-submission.csv --model claude-haiku

  # Grade with OpenAI o4-mini (reasoning model)
  python main.py grade exam1-submission.csv --model o4-mini

  # Grade with OpenAI GPT-4.1-mini
  python main.py grade exam1-submission.csv --model gpt-4.1-mini

  # Grade with DeepSeek-R1 (local Ollama model)
  python main.py grade exam1-submission.csv --model deepseek-r1

  # Grade with Llama 3.1 8B (local Ollama model)
  python main.py grade exam1-submission.csv --model llama3.1:8b

  # Generate overall statistics
  python main.py stats output/claude-claude-sonnet-4-6/grading_results.csv

  # Generate per-grader statistics
  python main.py grader-stats output/claude-claude-sonnet-4-6/grading_results.csv

  # Generate visualizations
  python main.py visualize output/claude-claude-sonnet-4-6/grading_results.csv

  # Grade first 50 submissions
  python main.py grade exam1-submission.csv --max-students 50 --model claude-sonnet

Output Structure:
  output/
  ├── gemini/
  ├── claude-claude-sonnet-4-6/
  ├── claude-claude-opus-4-6/
  ├── claude-claude-haiku-4-5-20251001/
  ├── openai-gpt-4-1-mini/
  ├── openai-o4-mini/
  ├── llama3-1-8b/
  └── deepseek-r1/

Environment Variables:
  GEMINI_API_KEY       Your Google Gemini API key
  OPENAI_API_KEY       Your OpenAI API key
  ANTHROPIC_API_KEY    Your Anthropic API key (required for Claude models)

Available Models:
  Claude:  claude-sonnet (recommended), claude-opus, claude-haiku
           Full names: claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5-20251001
  OpenAI:  o4-mini (reasoning), o3-mini, gpt-4.1-mini, gpt-4.1, gpt-4o-mini, gpt-4o
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
        default=0.0,
        help='Delay between sequential API calls in seconds (default: 0 — not needed with concurrent mode)'
    )
    grade_parser.add_argument(
        '--model',
        default='gemini',
        help='Model to use: "gemini", OpenAI model (e.g., "gpt-4o-mini"), or Ollama model (e.g., "llama3.1:8b") (default: gemini)'
    )
    grade_parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of concurrent API calls (default: 5; use 1 to disable concurrency)'
    )
    grade_parser.add_argument(
        '--no-calibration',
        action='store_true',
        default=False,
        help='Disable post-processing grade curve calibration (use raw LLM scores)'
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

    # Compare command
    cmp_parser = subparsers.add_parser(
        'compare', help='Compare ensemble vs single-model accuracy'
    )
    cmp_group = cmp_parser.add_mutually_exclusive_group(required=True)
    cmp_group.add_argument(
        '--models', nargs='+',
        choices=['gemini', 'o4-mini', 'claude', 'ensemble'],
        help='Model shortnames to compare (uses default output/ subfolders)'
    )
    cmp_group.add_argument(
        '--files', nargs='+',
        help='Explicit paths to grading_results.csv files'
    )
    cmp_parser.add_argument(
        '--labels', nargs='+',
        help='Display labels when using --files'
    )
    cmp_parser.add_argument(
        '--output-dir', default='output/comparison',
        help='Directory for report and charts (default: output/comparison)'
    )
    cmp_parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip chart generation'
    )

    args = parser.parse_args()
    
    if args.command == 'grade':
        success = grade_submissions(
            args.input_csv,
            args.output,
            args.max_students,
            args.rate_limit,
            args.model,
            calibration=not args.no_calibration,
            concurrent_workers=args.workers,
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

    elif args.command == 'compare':
        # Delegate entirely to ensemble_comparison module
        import sys
        from sqlAutograder import ensemble_comparison
        # Reconstruct argv for ensemble_comparison.main()
        cmp_args = ['ensemble_comparison.py']
        if args.models:
            cmp_args += ['--models'] + args.models
        else:
            cmp_args += ['--files'] + args.files
            if args.labels:
                cmp_args += ['--labels'] + args.labels
        cmp_args += ['--output-dir', args.output_dir]
        if args.no_plots:
            cmp_args.append('--no-plots')
        sys.argv = cmp_args
        ensemble_comparison.main()
        exit(0)
    
    else:
        parser.print_help()
        exit(1)


if __name__ == '__main__':
    main()