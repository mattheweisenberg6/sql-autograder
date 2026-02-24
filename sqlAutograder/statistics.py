"""
Statistics generation module for analyzing grading results.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List


class GradingStatistics:
    """Generates statistics comparing LLM and human grading."""
    
    def __init__(self, csv_path: str):
        """
        Initialize statistics generator.
        
        Args:
            csv_path: Path to the results CSV file
        """
        self.csv_path = csv_path
        self.df: pd.DataFrame = None
        self.valid_df: pd.DataFrame = None
    
    def _extract_model_name(self) -> str:
        """Extract model name from folder path."""
        # Get parent folder name (e.g., output/llama3-1-8b/grading_results.csv -> llama3-1-8b)
        parent_folder = os.path.basename(os.path.dirname(self.csv_path))
        if parent_folder and parent_folder != "output":
            return parent_folder.replace("-", " ").replace("_", " ").title()
        return "LLM"
    
    def load_and_validate(self) -> bool:
        """
        Load results and filter to valid comparisons.
        
        Returns:
            bool: True if data loaded successfully
        """
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Filter to valid results (LLM score >= 0 means grading succeeded)
            self.valid_df = self.df[self.df['q4_1_llm_score'] >= 0].copy()
            
            # Drop rows with missing grader scores
            required_cols = [
                'q4_1_grader_score', 'q4_2_grader_score', 
                'q4_3_grader_score', 'q4_4_grader_score', 
                'q4_5_grader_score'
            ]
            self.valid_df = self.valid_df.dropna(subset=required_cols)
            
            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def assign_grader(self, index: int) -> str:
        """
        Assign grader based on student index (0-based).
        
        Grader assignment:
        - G1: Students 0-54 (indices 0-54)
        - G2: Students 55-109 (indices 55-109)
        - G3: Students 110-164 (indices 110-164)
        - G4: Students 165-219 (indices 165-219)
        - G5: Students 220-274 (indices 220-274)
        - G6: Students 275+ (indices 275+)
        
        Args:
            index: 0-based row index in the dataframe
            
        Returns:
            Grader ID (G1-G6)
        """
        if 0 <= index < 55:
            return 'G1'
        elif 55 <= index < 110:
            return 'G2'
        elif 110 <= index < 165:
            return 'G3'
        elif 165 <= index < 220:
            return 'G4'
        elif 220 <= index < 275:
            return 'G5'
        else:
            return 'G6'
    
    def get_summary_statistics(self) -> Dict:
        """
        Calculate summary statistics across all students.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.valid_df is None or len(self.valid_df) == 0:
            return {}
        
        total_llm = self.valid_df['total_llm_score']
        total_human = self.valid_df['total_grader_score']
        total_diff = self.valid_df['total_score_difference']

        exact_matches = len(total_diff[abs(total_diff) < 0.5])
        agreement_rate = (exact_matches / len(self.valid_df)) * 100

        # Question-level agreement
        questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
        question_agreement = {}
        
        for q in questions:
            q_col = f'q{q.replace(".", "_")}_score_difference'
            q_diff = self.valid_df[q_col]
            q_exact = len(q_diff[abs(q_diff) < 0.1])
            question_agreement[f'Q{q}'] = (q_exact / len(self.valid_df)) * 100
        
        return {
            'total_students': len(self.df),
            'valid_students': len(self.valid_df),
            'human_avg': total_human.mean(),
            'human_std': total_human.std(),
            'llm_avg': total_llm.mean(),
            'llm_std': total_llm.std(),
            'avg_difference': total_diff.mean(),
            'overall_agreement': agreement_rate,
            'question_agreement': question_agreement
        }
    
    def get_per_question_stats(self) -> Dict[str, Dict]:
        """
        Calculate statistics for each question.
        
        Returns:
            Dictionary mapping question numbers to their statistics
        """
        if self.valid_df is None:
            return {}
        
        questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
        stats = {}
        
        for q in questions:
            q_prefix = f'q{q.replace(".", "_")}'
            
            llm_scores = self.valid_df[f'{q_prefix}_llm_score']
            human_scores = self.valid_df[f'{q_prefix}_grader_score']
            differences = self.valid_df[f'{q_prefix}_score_difference']
            
            exact_matches = len(differences[abs(differences) < 0.1])
            llm_higher = len(differences[differences > 0.1])
            llm_lower = len(differences[differences < -0.1])
            
            stats[q] = {
                'llm_avg': llm_scores.mean(),
                'llm_std': llm_scores.std(),
                'human_avg': human_scores.mean(),
                'human_std': human_scores.std(),
                'avg_difference': differences.mean(),
                'exact_matches': exact_matches,
                'agreement_rate': (exact_matches / len(self.valid_df)) * 100,
                'llm_higher': llm_higher,
                'llm_lower': llm_lower
            }
        
        return stats
    
    def get_per_grader_stats(self) -> Dict[str, Dict]:
        """
        Calculate statistics for each grader (G1-G6).
        
        Returns:
            Dictionary mapping grader IDs to their statistics
        """
        if self.valid_df is None:
            return {}
        
        # Assign graders based on index
        self.valid_df['grader'] = self.valid_df.index.map(self.assign_grader)
        
        questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
        grader_stats = {}
        
        for grader in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']:
            grader_data = self.valid_df[self.valid_df['grader'] == grader]
            
            if len(grader_data) == 0:
                continue
            
            # Total scores
            total_llm = grader_data['total_llm_score']
            total_human = grader_data['total_grader_score']
            total_diff = grader_data['total_score_difference']
            
            # Per-question stats for this grader
            question_stats = {}
            for q in questions:
                q_prefix = f'q{q.replace(".", "_")}'
                llm_scores = grader_data[f'{q_prefix}_llm_score']
                human_scores = grader_data[f'{q_prefix}_grader_score']
                differences = grader_data[f'{q_prefix}_score_difference']
                
                exact_matches = len(differences[abs(differences) < 0.1])
                
                question_stats[q] = {
                    'llm_avg': llm_scores.mean(),
                    'llm_std': llm_scores.std(),
                    'human_avg': human_scores.mean(),
                    'human_std': human_scores.std(),
                    'avg_difference': differences.mean(),
                    'agreement_rate': (exact_matches / len(grader_data)) * 100 if len(grader_data) > 0 else 0
                }
            
            grader_stats[grader] = {
                'num_students': len(grader_data),
                'total_llm_avg': total_llm.mean(),
                'total_llm_std': total_llm.std(),
                'total_human_avg': total_human.mean(),
                'total_human_std': total_human.std(),
                'total_avg_diff': total_diff.mean(),
                'questions': question_stats
            }
        
        return grader_stats
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive text report.
        
        Returns:
            Formatted statistics report as string
        """
        if not self.load_and_validate():
            return "Error: Could not load or validate data"
        
        model_name = self._extract_model_name()
        summary = self.get_summary_statistics()
        per_question = self.get_per_question_stats()
        
        report = []
        report.append("=" * 80)
        report.append(f"SQL AUTOGRADING STATISTICS REPORT - {model_name}")
        report.append("=" * 80)
        report.append("")
        
        # Summary section
        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total students: {summary['total_students']}")
        report.append(f"Valid comparisons: {summary['valid_students']}")
        report.append(f"Human average: {summary['human_avg']:.1f}±{summary['human_std']:.1f}")
        report.append(f"{model_name} average: {summary['llm_avg']:.1f}±{summary['llm_std']:.1f}")
        report.append(f"Average difference: {summary['avg_difference']:.1f}")
        report.append(f"Overall agreement: {summary['overall_agreement']:.1f}%")
        report.append("")
        
        # Question-level agreement
        report.append("Question-level agreement:")
        for q_name, agree_rate in summary['question_agreement'].items():
            report.append(f"  {q_name}: {agree_rate:.1f}%")
        report.append("")
        
        # Per-question details
        report.append("=" * 80)
        report.append("PER-QUESTION STATISTICS")
        report.append("=" * 80)
        report.append("")
        
        for q_num, stats in per_question.items():
            report.append(f"QUESTION {q_num} (Out of 10)")
            report.append("-" * 80)
            report.append(f"Average {model_name} score: {stats['llm_avg']:.1f}±{stats['llm_std']:.1f}")
            report.append(f"Average Human score: {stats['human_avg']:.1f}±{stats['human_std']:.1f}")
            report.append(f"Average difference: {stats['avg_difference']:.1f}")
            report.append(f"Agreement rate: {stats['agreement_rate']:.1f}%")
            report.append(f"{model_name} scored higher: {stats['llm_higher']} ({stats['llm_higher']/len(self.valid_df)*100:.1f}%)")
            report.append(f"{model_name} scored lower: {stats['llm_lower']} ({stats['llm_lower']/len(self.valid_df)*100:.1f}%)")
            report.append("")
        
        return "\n".join(report)
    
    def generate_per_grader_report(self) -> str:
        """
        Generate a per-grader statistics report.
        
        Returns:
            Formatted per-grader statistics report as string
        """
        if not self.load_and_validate():
            return "Error: Could not load or validate data"
        
        model_name = self._extract_model_name()
        grader_stats = self.get_per_grader_stats()
        
        report = []
        report.append("=" * 80)
        report.append(f"PER-GRADER STATISTICS REPORT - {model_name}")
        report.append("=" * 80)
        report.append("")
        report.append("Grader Assignments:")
        report.append("  G1: Students 1-55")
        report.append("  G2: Students 56-110")
        report.append("  G3: Students 111-165")
        report.append("  G4: Students 166-220")
        report.append("  G5: Students 221-275")
        report.append("  G6: Students 276-330")
        report.append("")
        
        # Overall table
        report.append("=" * 80)
        report.append("TOTAL SCORES (Out of 50)")
        report.append("=" * 80)
        report.append("")
        report.append(f"Grader | #Students | Human: avg±std | {model_name}: avg±std   | Avg Diff")
        report.append("-" * 80)
        
        for grader in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']:
            if grader not in grader_stats:
                continue
            
            stats = grader_stats[grader]
            report.append(
                f"{grader:>6} | {stats['num_students']:>9} | "
                f"{stats['total_human_avg']:>5.1f}±{stats['total_human_std']:>4.1f}   | "
                f"{stats['total_llm_avg']:>5.1f}±{stats['total_llm_std']:>4.1f}   | "
                f"{stats['total_avg_diff']:>+8.1f}"
            )
        
        report.append("")
        
        # Per-question breakdown for each grader
        questions = ['4.1', '4.2', '4.3', '4.4', '4.5']
        
        for q in questions:
            report.append("=" * 80)
            report.append(f"QUESTION {q} (Out of 10)")
            report.append("=" * 80)
            report.append("")
            report.append(f"Grader |  N  | Human: avg±std | {model_name}: avg±std | Avg Diff | Agreement")
            report.append("-" * 80)
            
            for grader in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']:
                if grader not in grader_stats:
                    continue
                
                stats = grader_stats[grader]
                q_stats = stats['questions'][q]
                
                report.append(
                    f"{grader:>6} | {stats['num_students']:>3} | "
                    f"{q_stats['human_avg']:>4.1f}±{q_stats['human_std']:>3.1f}     | "
                    f"{q_stats['llm_avg']:>4.1f}±{q_stats['llm_std']:>3.1f}    | "
                    f"{q_stats['avg_difference']:>+8.1f} | "
                    f"{q_stats['agreement_rate']:>8.1f}%"
                )
            
            report.append("")
        
        return "\n".join(report)
    
    def save_report(self, output_path: str) -> bool:
        """
        Generate and save statistics report to file.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            bool: True if successful
        """
        try:
            report = self.generate_report()
            with open(output_path, 'w') as f:
                f.write(report)
            return True
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
    
    def save_per_grader_report(self, output_path: str) -> bool:
        """
        Generate and save per-grader statistics report to file.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            bool: True if successful
        """
        try:
            report = self.generate_per_grader_report()
            with open(output_path, 'w') as f:
                f.write(report)
            return True
        except Exception as e:
            print(f"Error saving per-grader report: {e}")
            return False