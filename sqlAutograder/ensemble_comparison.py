"""
Ensemble vs Single-Model Comparison Report.

Loads grading_results.csv files from multiple model output folders and compares
them against each other and against human grader scores.

Metrics reported per model:
  - Mean Absolute Error (MAE) vs human — primary accuracy metric
  - Root Mean Squared Error (RMSE) vs human
  - Mean signed difference (bias: negative = model grades lower than human)
  - Exact agreement rate (score matches human within ±0 points)
  - Within-1 agreement rate (score within ±1 of human)
  - Per-question breakdown of all the above

Usage:
    python ensemble_comparison.py \\
        --models gemini o4-mini claude ensemble \\
        --output-dir output/comparison

    # Or compare specific result files directly:
    python ensemble_comparison.py \\
        --files output/gemini/grading_results.csv \\
                output/openai-o4-mini/grading_results.csv \\
                output/claude-claude-sonnet-4-6/grading_results.csv \\
                output/ensemble/grading_results.csv
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


QUESTIONS = ['4.1', '4.2', '4.3', '4.4', '4.5']
Q_PREFIXES = [f'q{q.replace(".", "_")}' for q in QUESTIONS]

# Default model-name → folder mapping (under output/)
_DEFAULT_FOLDERS = {
    'gemini':   'gemini',
    'o4-mini':  'openai-o4-mini',
    'claude':   'claude-claude-sonnet-4-6',
    'ensemble': 'ensemble',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(csv_path: str, model_label: str) -> Optional[pd.DataFrame]:
    """Load a grading results CSV and filter to valid rows."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ✗ Not found: {csv_path}")
        return None
    except Exception as e:
        print(f"  ✗ Error loading {csv_path}: {e}")
        return None

    # Keep only rows where grading succeeded (llm_score >= 0)
    valid = df[df['q4_1_llm_score'] >= 0].copy()

    required = [f'{p}_grader_score' for p in Q_PREFIXES]
    valid = valid.dropna(subset=required)

    if len(valid) == 0:
        print(f"  ✗ No valid rows in {csv_path}")
        return None

    valid['_model'] = model_label
    print(f"  ✓ {model_label}: {len(valid)} valid students (of {len(df)} total)")
    return valid


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, label: str) -> Dict:
    """Compute accuracy metrics for one model's results vs human grader."""
    metrics = {'model': label, 'n': len(df)}

    total_diffs = []
    q_metrics = {}

    for q, prefix in zip(QUESTIONS, Q_PREFIXES):
        llm   = df[f'{prefix}_llm_score'].values.astype(float)
        human = df[f'{prefix}_grader_score'].values.astype(float)
        diff  = llm - human

        mae   = np.mean(np.abs(diff))
        rmse  = np.sqrt(np.mean(diff ** 2))
        bias  = np.mean(diff)
        exact = np.mean(np.abs(diff) < 0.5) * 100          # within ±0
        within1 = np.mean(np.abs(diff) <= 1.0) * 100       # within ±1

        q_metrics[q] = {
            'mae': mae, 'rmse': rmse, 'bias': bias,
            'exact': exact, 'within1': within1,
        }
        total_diffs.append(diff)

    # Total score metrics
    total_llm   = df['total_llm_score'].values.astype(float)
    total_human = df['total_grader_score'].values.astype(float)
    total_diff  = total_llm - total_human

    metrics['total_mae']     = np.mean(np.abs(total_diff))
    metrics['total_rmse']    = np.sqrt(np.mean(total_diff ** 2))
    metrics['total_bias']    = np.mean(total_diff)
    metrics['total_exact']   = np.mean(np.abs(total_diff) < 0.5) * 100
    metrics['total_within1'] = np.mean(np.abs(total_diff) <= 1.0) * 100
    metrics['questions']     = q_metrics

    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_text_report(all_metrics: List[Dict]) -> str:
    """Generate a formatted text comparison report."""
    lines = []
    sep = '=' * 90

    lines += [sep, 'ENSEMBLE vs SINGLE-MODEL COMPARISON REPORT', sep, '']

    # ---- Summary table ----
    lines += ['TOTAL SCORE COMPARISON (out of 50)', '-' * 90]
    header = f"{'Model':<22} {'N':>4}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>7}  {'Exact%':>7}  {'Within1%':>9}"
    lines.append(header)
    lines.append('-' * 90)

    # Sort by MAE ascending (best first)
    for m in sorted(all_metrics, key=lambda x: x['total_mae']):
        line = (
            f"{m['model']:<22} {m['n']:>4}  "
            f"{m['total_mae']:>6.2f}  {m['total_rmse']:>6.2f}  "
            f"{m['total_bias']:>+7.2f}  {m['total_exact']:>6.1f}%  "
            f"{m['total_within1']:>8.1f}%"
        )
        lines.append(line)
    lines.append('')

    # ---- Per-question breakdown ----
    lines += [sep, 'PER-QUESTION BREAKDOWN (MAE vs human grader)', sep, '']

    for q in QUESTIONS:
        lines += [f'Question {q}', '-' * 90]
        lines.append(
            f"{'Model':<22} {'MAE':>6}  {'RMSE':>6}  {'Bias':>7}  {'Exact%':>7}  {'Within1%':>9}"
        )
        lines.append('-' * 90)

        for m in sorted(all_metrics, key=lambda x: x['questions'][q]['mae']):
            qm = m['questions'][q]
            line = (
                f"{m['model']:<22} "
                f"{qm['mae']:>6.2f}  {qm['rmse']:>6.2f}  "
                f"{qm['bias']:>+7.2f}  {qm['exact']:>6.1f}%  "
                f"{qm['within1']:>8.1f}%"
            )
            lines.append(line)
        lines.append('')

    # ---- Winner summary ----
    lines += [sep, 'WINNER SUMMARY (lowest MAE = best)', '-' * 90]
    best_total = min(all_metrics, key=lambda x: x['total_mae'])
    lines.append(f"Overall best model: {best_total['model']}  (MAE={best_total['total_mae']:.2f})")
    lines.append('')

    for q in QUESTIONS:
        best_q = min(all_metrics, key=lambda x: x['questions'][q]['mae'])
        lines.append(f"  Q{q} best: {best_q['model']}  (MAE={best_q['questions'][q]['mae']:.2f})")

    lines.append('')
    lines.append(sep)
    lines.append('METRIC DEFINITIONS')
    lines.append('-' * 90)
    lines.append('MAE      Mean Absolute Error vs human grader (lower = better)')
    lines.append('RMSE     Root Mean Squared Error (penalises large misses more than MAE)')
    lines.append('Bias     Mean signed difference LLM−Human (negative = model grades lower)')
    lines.append('Exact%   % of scores matching human within ±0 points')
    lines.append('Within1% % of scores matching human within ±1 point')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _model_colors(models: List[str]) -> Dict[str, str]:
    palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
               '#937860', '#DA8BC3', '#8C8C8C']
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def plot_mae_comparison(all_metrics: List[Dict], output_path: str):
    """Bar chart: MAE per model, grouped by question + total."""
    labels = [m['model'] for m in all_metrics]
    colors = _model_colors(labels)
    categories = [f'Q{q}' for q in QUESTIONS] + ['Total']

    x = np.arange(len(categories))
    n = len(labels)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, m in enumerate(all_metrics):
        maes = [m['questions'][q]['mae'] for q in QUESTIONS] + [m['total_mae']]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, maes, width, label=m['model'],
                      color=colors[m['model']], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Mean Absolute Error (lower = better)', fontsize=11)
    ax.set_title('MAE vs Human Grader — Ensemble vs Single Models', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ MAE comparison chart → {output_path}")


def plot_bias_comparison(all_metrics: List[Dict], output_path: str):
    """Bar chart: signed bias per model per question. Shows over/under grading."""
    labels = [m['model'] for m in all_metrics]
    colors = _model_colors(labels)
    categories = [f'Q{q}' for q in QUESTIONS] + ['Total']

    x = np.arange(len(categories))
    n = len(labels)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    for i, m in enumerate(all_metrics):
        biases = [m['questions'][q]['bias'] for q in QUESTIONS] + [m['total_bias']]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, biases, width, label=m['model'],
                      color=colors[m['model']], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, biases):
            ypos = bar.get_height() + (0.05 if val >= 0 else -0.15)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{val:+.2f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Bias: LLM − Human (negative = grades lower)', fontsize=11)
    ax.set_title('Grading Bias — Ensemble vs Single Models', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Bias comparison chart → {output_path}")


def plot_agreement_comparison(all_metrics: List[Dict], output_path: str):
    """Grouped bar: exact agreement % and within-1 % per model."""
    labels = [m['model'] for m in all_metrics]
    colors = _model_colors(labels)
    x = np.arange(len(labels))
    width = 0.35

    exact   = [m['total_exact']   for m in all_metrics]
    within1 = [m['total_within1'] for m in all_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, exact,   width, label='Exact match (±0)',  color='#4C72B0', edgecolor='white')
    b2 = ax.bar(x + width / 2, within1, width, label='Within ±1 point',  color='#55A868', edgecolor='white')

    for bar, val in list(zip(b1, exact)) + list(zip(b2, within1)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Agreement with Human Grader (%)', fontsize=11)
    ax.set_title('Score Agreement — Ensemble vs Single Models (Total Score)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Agreement comparison chart → {output_path}")


def plot_per_question_within1(all_metrics: List[Dict], output_path: str):
    """Heatmap: within-1 agreement % per model × question."""
    labels = [m['model'] for m in all_metrics]
    q_labels = [f'Q{q}' for q in QUESTIONS]

    matrix = np.array([
        [m['questions'][q]['within1'] for q in QUESTIONS]
        for m in all_metrics
    ])

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.9 + 1)))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=40, vmax=100, aspect='auto')

    ax.set_xticks(range(len(q_labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(q_labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=11)

    for i in range(len(labels)):
        for j in range(len(QUESTIONS)):
            val = matrix[i, j]
            color = 'white' if val < 55 or val > 85 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Within-±1 Agreement (%)', rotation=270, labelpad=18, fontsize=10)
    ax.set_title('Per-Question Within-±1 Agreement vs Human Grader', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Per-question heatmap → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare ensemble vs single-model grading accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all four models using default folder names under output/
  python ensemble_comparison.py --models gemini o4-mini claude ensemble

  # Compare using explicit CSV paths
  python ensemble_comparison.py \\
      --files output/gemini/grading_results.csv \\
              output/openai-o4-mini/grading_results.csv \\
              output/claude-claude-sonnet-4-6/grading_results.csv \\
              output/ensemble/grading_results.csv \\
      --labels Gemini o4-mini Claude Ensemble

  # Compare just two models
  python ensemble_comparison.py --models claude ensemble
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--models', nargs='+',
        choices=list(_DEFAULT_FOLDERS.keys()),
        help='Model shortnames to compare (uses default output/ subfolder names)'
    )
    group.add_argument(
        '--files', nargs='+',
        help='Explicit paths to grading_results.csv files'
    )
    parser.add_argument(
        '--labels', nargs='+',
        help='Display labels for --files mode (must match number of files)'
    )
    parser.add_argument(
        '--output-dir', default='output/comparison',
        help='Directory to save report and charts (default: output/comparison)'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip chart generation (text report only)'
    )

    args = parser.parse_args()

    # Resolve file paths and labels
    if args.models:
        paths  = [f'output/{_DEFAULT_FOLDERS[m]}/grading_results.csv' for m in args.models]
        labels = args.models
    else:
        paths  = args.files
        if args.labels:
            if len(args.labels) != len(paths):
                parser.error('--labels must have the same number of entries as --files')
            labels = args.labels
        else:
            labels = [Path(p).parent.name for p in paths]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('ENSEMBLE COMPARISON')
    print('=' * 60)
    print()
    print('Loading result files...')

    all_metrics = []
    for path, label in zip(paths, labels):
        df = load_results(path, label)
        if df is not None:
            all_metrics.append(compute_metrics(df, label))

    if len(all_metrics) < 2:
        print('\n✗ Need at least 2 valid result files to compare.')
        return

    print()

    # Text report
    report = generate_text_report(all_metrics)
    print(report)

    report_path = output_dir / 'comparison_report.txt'
    report_path.write_text(report, encoding='utf-8')
    print(f'\n✓ Report saved → {report_path}')

    # Charts
    if not args.no_plots:
        print('\nGenerating charts...')
        plot_mae_comparison(all_metrics,        str(output_dir / 'comparison_mae.png'))
        plot_bias_comparison(all_metrics,       str(output_dir / 'comparison_bias.png'))
        plot_agreement_comparison(all_metrics,  str(output_dir / 'comparison_agreement.png'))
        plot_per_question_within1(all_metrics,  str(output_dir / 'comparison_heatmap.png'))

    print(f'\n✓ All outputs saved to: {output_dir}/')


if __name__ == '__main__':
    main()