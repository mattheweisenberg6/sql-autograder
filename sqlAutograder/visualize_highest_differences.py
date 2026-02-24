#!/usr/bin/env python3
"""
Visualize the highest differences between human grader and LLM scores per question.
Shows the top cases where LLM and human graders disagreed the most.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def load_and_filter_data(csv_path: str) -> pd.DataFrame:
    """
    Load results CSV and filter to valid comparisons.
    
    Args:
        csv_path: Path to the grading results CSV file
        
    Returns:
        Filtered DataFrame with valid comparisons
    """
    df = pd.read_csv(csv_path)
    
    # Filter to valid results (LLM score >= 0 means grading succeeded)
    valid_df = df[df['q4_1_llm_score'] >= 0].copy()
    
    # Drop rows with missing grader scores
    required_cols = [
        'q4_1_grader_score', 'q4_2_grader_score', 
        'q4_3_grader_score', 'q4_4_grader_score', 
        'q4_5_grader_score'
    ]
    valid_df = valid_df.dropna(subset=required_cols)
    
    return valid_df


def find_highest_differences(df: pd.DataFrame, top_n: int = 10):
    """
    Find the students with the highest score differences for each question.
    
    Args:
        df: DataFrame with grading results
        top_n: Number of top differences to find per question
        
    Returns:
        Dictionary mapping question numbers to DataFrames with top differences
    """
    questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
    highest_diffs = {}
    
    for q in questions:
        q_prefix = f'q{q.replace(".", "_")}'
        diff_col = f'{q_prefix}_score_difference'
        
        # Get absolute differences and sort
        q_data = df[['student_id', 'student_name', 
                     f'{q_prefix}_grader_score', 
                     f'{q_prefix}_llm_score',
                     diff_col]].copy()
        
        q_data['abs_difference'] = abs(q_data[diff_col])
        q_data = q_data.sort_values('abs_difference', ascending=False).head(top_n)
        
        highest_diffs[q] = q_data
    
    return highest_diffs


def plot_highest_differences_grid(df: pd.DataFrame, output_path: str = None, top_n: int = 10):
    """
    Create a grid visualization showing the highest differences for each question.
    
    Args:
        df: DataFrame with grading results
        output_path: Path to save the figure
        top_n: Number of top differences to show per question
    """
    questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
    
    # Create figure with 5 subplots (one per question)
    fig, axes = plt.subplots(5, 1, figsize=(14, 20))
    
    for idx, q in enumerate(questions):
        ax = axes[idx]
        q_prefix = f'q{q.replace(".", "_")}'
        
        # Get data for this question
        q_data = df[['student_id', 'student_name',
                     f'{q_prefix}_grader_score',
                     f'{q_prefix}_llm_score',
                     f'{q_prefix}_score_difference']].copy()
        
        q_data['abs_difference'] = abs(q_data[f'{q_prefix}_score_difference'])
        q_data = q_data.sort_values('abs_difference', ascending=False).head(top_n)
        
        # Create labels for x-axis (student IDs)
        labels = [f"{row['student_id']}" for _, row in q_data.iterrows()]
        x = np.arange(len(labels))
        
        human_scores = q_data[f'{q_prefix}_grader_score'].values
        llm_scores = q_data[f'{q_prefix}_llm_score'].values
        differences = q_data[f'{q_prefix}_score_difference'].values
        
        # Create grouped bar chart
        width = 0.35
        bars1 = ax.bar(x - width/2, human_scores, width, label='Human Score', 
                       color='#FF6B6B', edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x + width/2, llm_scores, width, label='LLM Score',
                       color='#4ECDC4', edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for i, (h_score, l_score, diff) in enumerate(zip(human_scores, llm_scores, differences)):
            # Human score label
            ax.text(i - width/2, h_score + 0.3, f'{h_score:.0f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            # LLM score label
            ax.text(i + width/2, l_score + 0.3, f'{l_score:.0f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Difference annotation
            y_pos = max(h_score, l_score) + 1.5
            color = 'green' if diff > 0 else 'red' if diff < 0 else 'gray'
            ax.text(i, y_pos, f'Δ={diff:+.0f}',
                   ha='center', va='bottom', fontsize=9, 
                   fontweight='bold', color=color)
        
        # Formatting
        ax.set_xlabel('Student ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score (out of 10)', fontsize=11, fontweight='bold')
        ax.set_title(f'Question {q} - Top {top_n} Highest Score Differences', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add stats box
        max_diff = q_data['abs_difference'].max()
        avg_diff = q_data['abs_difference'].mean()
        stats_text = f'Max |Diff|: {max_diff:.1f}\nAvg |Diff|: {avg_diff:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Highest Score Differences by Question (Human vs LLM)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Highest differences visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_highest_differences_heatmap(df: pd.DataFrame, output_path: str = None, top_n: int = 15):
    """
    Create a heatmap showing the score differences for top students across all questions.
    
    Args:
        df: DataFrame with grading results
        output_path: Path to save the figure
        top_n: Number of students to include
    """
    # Calculate total absolute difference for each student
    questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
    df_copy = df.copy()
    
    total_abs_diff = 0
    for q in questions:
        q_prefix = f'q{q.replace(".", "_")}'
        df_copy[f'{q_prefix}_abs_diff'] = abs(df_copy[f'{q_prefix}_score_difference'])
        total_abs_diff += df_copy[f'{q_prefix}_abs_diff']
    
    df_copy['total_abs_diff'] = total_abs_diff
    
    # Get top N students with highest total absolute differences
    top_students = df_copy.nlargest(top_n, 'total_abs_diff')
    
    # Create matrix for heatmap
    heatmap_data = []
    student_labels = []
    
    for _, row in top_students.iterrows():
        student_labels.append(f"{row['student_id']}")
        diffs = [row[f'q{q.replace(".", "_")}_score_difference'] for q in questions]
        heatmap_data.append(diffs)
    
    heatmap_matrix = np.array(heatmap_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use diverging colormap centered at 0
    im = ax.imshow(heatmap_matrix, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(questions)))
    ax.set_yticks(np.arange(len(student_labels)))
    ax.set_xticklabels([f'Q{q}' for q in questions], fontsize=12, fontweight='bold')
    ax.set_yticklabels(student_labels, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score Difference (LLM - Human)', rotation=270, labelpad=20, 
                   fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(student_labels)):
        for j in range(len(questions)):
            value = heatmap_matrix[i, j]
            color = 'white' if abs(value) > 5 else 'black'
            text = ax.text(j, i, f'{value:+.0f}',
                          ha="center", va="center", color=color,
                          fontsize=10, fontweight='bold')
    
    ax.set_title(f'Top {top_n} Students with Highest Total Score Differences\n(LLM - Human)',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Question', fontsize=12, fontweight='bold')
    ax.set_ylabel('Student ID', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_path: str = None):
    """
    Generate a text summary of the highest differences per question.
    
    Args:
        df: DataFrame with grading results
        output_path: Path to save the report
    """
    questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
    
    report = []
    report.append("=" * 80)
    report.append("HIGHEST SCORE DIFFERENCES SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    for q in questions:
        q_prefix = f'q{q.replace(".", "_")}'
        
        # Get data for this question
        q_data = df[[f'{q_prefix}_score_difference']].copy()
        q_data['abs_diff'] = abs(q_data[f'{q_prefix}_score_difference'])
        
        max_diff_idx = q_data['abs_diff'].idxmax()
        max_diff_row = df.loc[max_diff_idx]
        
        max_diff = max_diff_row[f'{q_prefix}_score_difference']
        max_abs_diff = abs(max_diff)
        human_score = max_diff_row[f'{q_prefix}_grader_score']
        llm_score = max_diff_row[f'{q_prefix}_llm_score']
        student_id = max_diff_row['student_id']
        
        # Overall stats for this question
        all_diffs = df[f'{q_prefix}_score_difference']
        avg_diff = all_diffs.mean()
        std_diff = all_diffs.std()
        avg_abs_diff = abs(all_diffs).mean()
        
        report.append(f"QUESTION {q}")
        report.append("-" * 80)
        report.append(f"Highest absolute difference: {max_abs_diff:.1f} points")
        report.append(f"  Student ID: {student_id}")
        report.append(f"  Human score: {human_score:.1f}")
        report.append(f"  LLM score: {llm_score:.1f}")
        report.append(f"  Difference: {max_diff:+.1f} (LLM - Human)")
        report.append("")
        report.append(f"Overall statistics for Question {q}:")
        report.append(f"  Mean difference: {avg_diff:+.1f}")
        report.append(f"  Std deviation: {std_diff:.1f}")
        report.append(f"  Mean absolute difference: {avg_abs_diff:.1f}")
        report.append("")
    
    report_text = "\n".join(report)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Summary report saved to: {output_path}")
    else:
        print(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Visualize highest differences between human and LLM scores per question'
    )
    parser.add_argument('csv_path', type=str, 
                       help='Path to the grading results CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output files (default: same as CSV)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top differences to show per question (default: 10)')
    parser.add_argument('--no-grid', action='store_true',
                       help='Skip grid visualization')
    parser.add_argument('--no-heatmap', action='store_true',
                       help='Skip heatmap visualization')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip text summary report')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.csv_path}")
    df = load_and_filter_data(args.csv_path)
    print(f"Loaded {len(df)} valid student submissions")
    print()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.csv_path).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model name for filenames
    parent_folder = Path(args.csv_path).parent.name
    if parent_folder and parent_folder != "output":
        model_suffix = parent_folder
    else:
        model_suffix = "llm"
    
    # Generate visualizations
    if not args.no_grid:
        grid_path = output_dir / f"highest_differences_grid_{model_suffix}.png"
        print("Generating grid visualization...")
        plot_highest_differences_grid(df, str(grid_path), args.top_n)
        print()
    
    if not args.no_heatmap:
        heatmap_path = output_dir / f"highest_differences_heatmap_{model_suffix}.png"
        print("Generating heatmap visualization...")
        plot_highest_differences_heatmap(df, str(heatmap_path), args.top_n + 5)
        print()
    
    if not args.no_report:
        report_path = output_dir / f"highest_differences_report_{model_suffix}.txt"
        print("Generating summary report...")
        generate_summary_report(df, str(report_path))
        print()
    
    print("✓ All visualizations complete!")


if __name__ == "__main__":
    main()