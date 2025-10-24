#!/usr/bin/env python3
"""
Inverse Problem Results Visualization
=====================================

This script analyzes and visualizes the inverse problem results from notch parameter
estimation using multi-fidelity surrogate models.

Author: Claude Code Assistant
Date: 2025-10-19
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Set style for academic publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class InverseResultsAnalyzer:
    """Analyzes inverse problem results and generates publication-quality plots."""

    def __init__(self, results_file: str):
        """
        Initialize analyzer with results file path.

        Parameters:
        -----------
        results_file : str
            Path to the inverseresults.txt file
        """
        self.results_file = Path(results_file)
        self.data = []
        self.df = None
        self.parse_results()

    def parse_results(self):
        """Parse the inverse results from the text file."""
        print("Parsing inverse problem results...")

        with open(self.results_file, 'r') as f:
            content = f.read()

        # Split into test cases
        test_cases = re.split(r'\n\d+\.\n', content)
        test_cases = [case for case in test_cases if case.strip()]

        for i, case in enumerate(test_cases, 1):
            try:
                # Extract true and estimated values for each parameter
                result = self._extract_test_case(case, i)
                if result:
                    self.data.append(result)
            except Exception as e:
                print(f"Warning: Could not parse test case {i}: {e}")

        # Convert to DataFrame for easier manipulation
        self.df = pd.DataFrame(self.data)
        print(f"Successfully parsed {len(self.df)} test cases")

        # Calculate derived metrics
        self._calculate_metrics()

    def _extract_test_case(self, case_text: str, case_id: int) -> Optional[Dict]:
        """Extract parameter values from a single test case."""
        result = {'test_id': case_id}

        # Extract notch_x values
        x_match = re.search(r'notch_x:.*?True:\s*([\d.]+).*?Estimated:\s*([\d.]+)', case_text, re.DOTALL)
        if x_match:
            result['notch_x_true'] = float(x_match.group(1))
            result['notch_x_est'] = float(x_match.group(2))

        # Extract notch_depth values
        depth_match = re.search(r'notch_depth:.*?True:\s*([\d.]+).*?Estimated:\s*([\d.]+)', case_text, re.DOTALL)
        if depth_match:
            result['notch_depth_true'] = float(depth_match.group(1))
            result['notch_depth_est'] = float(depth_match.group(2))

        # Extract notch_width values
        width_match = re.search(r'notch_width:.*?True:\s*([\d.]+).*?Estimated:\s*([\d.]+)', case_text, re.DOTALL)
        if width_match:
            result['notch_width_true'] = float(width_match.group(1))
            result['notch_width_est'] = float(width_match.group(2))

        # Ensure all parameters were found
        if len(result) != 7:  # test_id + 3*2 (true/est pairs)
            return None

        return result

    def _calculate_metrics(self):
        """Calculate percentage errors and severity classifications."""
        print("Calculating error metrics and severity classifications...")

        # Percentage errors
        self.df['notch_x_error_pct'] = np.abs((self.df['notch_x_est'] - self.df['notch_x_true']) / self.df['notch_x_true']) * 100
        self.df['notch_depth_error_pct'] = np.abs((self.df['notch_depth_est'] - self.df['notch_depth_true']) / self.df['notch_depth_true']) * 100
        self.df['notch_width_error_pct'] = np.abs((self.df['notch_width_est'] - self.df['notch_width_true']) / self.df['notch_width_true']) * 100

        # Severity metrics (based on thesis: s = depth × width)
        self.df['severity_true'] = self.df['notch_depth_true'] * self.df['notch_width_true']
        self.df['severity_est'] = self.df['notch_depth_est'] * self.df['notch_width_est']

        # Severity classification
        self.df['severity_class'] = self._classify_severity(self.df['severity_true'])

        # Success metrics
        self.df['location_success'] = np.abs(self.df['notch_x_est'] - self.df['notch_x_true']) < 0.05  # 5cm tolerance
        self.df['severity_success'] = self._classify_severity(self.df['severity_est']) == self.df['severity_class']

    def _classify_severity(self, severity_values: pd.Series) -> pd.Series:
        """Classify severity based on thesis criteria."""
        categories = []
        for s in severity_values:
            if 1e-8 <= s <= 4e-7:
                categories.append('Mild')
            elif 4e-7 < s <= 8e-7:
                categories.append('Moderate')
            elif 8e-7 < s <= 1.2e-6:
                categories.append('Severe')
            else:
                categories.append('Undefined')
        return pd.Series(categories)

    def create_plot1_accuracy_scatter(self, figsize=(15, 5)):
        """
        Create Plot 1: Parameter estimation accuracy X=Y scatter plots.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)

        Returns:
        --------
        fig : matplotlib.figure.Figure
        """
        print("Creating Plot 1: Parameter estimation accuracy...")

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Color mapping for severity
        color_map = {'Mild': '#2E8B57', 'Moderate': '#FFB347', 'Severe': '#DC143C'}

        parameters = [
            ('notch_x', 'Notch Location $x$ (m)', axes[0]),
            ('notch_depth', 'Notch Depth $d$ (m)', axes[1]),
            ('notch_width', 'Notch Width $w$ (m)', axes[2])
        ]

        for param, label, ax in parameters:
            true_col = f'{param}_true'
            est_col = f'{param}_est'

            # Plot points colored by severity
            for severity in ['Mild', 'Moderate', 'Severe']:
                mask = self.df['severity_class'] == severity
                if mask.any():
                    ax.scatter(self.df.loc[mask, true_col],
                             self.df.loc[mask, est_col],
                             c=color_map[severity], label=severity, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

            # X=Y line
            min_val = min(self.df[true_col].min(), self.df[est_col].min())
            max_val = max(self.df[true_col].max(), self.df[est_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

            # R² calculation
            r2 = np.corrcoef(self.df[true_col], self.df[est_col])[0, 1]**2
            ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Labels and formatting
            ax.set_xlabel(f'True {label}')
            ax.set_ylabel(f'Estimated {label}')
            ax.set_title(param.replace('_', ' ').title(), fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Equal aspect ratio for better comparison
            ax.set_aspect('equal', adjustable='box')

        # Single legend for all subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

        plt.tight_layout()
        return fig

    def create_plot2_error_distribution(self, figsize=(12, 8)):
        """
        Create Plot 2: Error distribution for all parameters.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)

        Returns:
        --------
        fig : matplotlib.figure.Figure
        """
        print("Creating Plot 2: Error distribution analysis...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Prepare data for plotting
        error_data = []
        param_names = []

        for param in ['notch_x', 'notch_depth', 'notch_width']:
            error_col = f'{param}_error_pct'
            error_data.extend(self.df[error_col].values)
            param_names.extend([param.replace('_', ' ').title()] * len(self.df))

        error_df = pd.DataFrame({
            'Error (%)': error_data,
            'Parameter': param_names
        })

        # Box plot
        sns.boxplot(data=error_df, x='Parameter', y='Error (%)', ax=ax1, palette='Set2')
        ax1.set_title('Parameter Estimation Error Distribution', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Percentage Error (%)', fontsize=12)
        ax1.set_xlabel('Parameter', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Add statistical annotations
        for i, param in enumerate(['notch_x', 'notch_depth', 'notch_width']):
            error_col = f'{param}_error_pct'
            median = self.df[error_col].median()
            ax1.text(i, median + 1, f'{median:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Violin plot for detailed distribution
        sns.violinplot(data=error_df, x='Parameter', y='Error (%)', ax=ax2, palette='Set2')
        ax2.set_title('Detailed Error Distribution', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Percentage Error (%)', fontsize=12)
        ax2.set_xlabel('Parameter', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_plot3_severity_cumulative(self, figsize=(15, 6)):
        """
        Create Plot 3: Cumulative error by severity classification.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)

        Returns:
        --------
        fig : matplotlib.figure.Figure
        """
        print("Creating Plot 3: Severity-based cumulative error analysis...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Color mapping
        color_map = {'Mild': '#2E8B57', 'Moderate': '#FFB347', 'Severe': '#DC143C'}

        # Plot 1: Success rate by severity
        severity_success = self.df.groupby('severity_class').agg({
            'location_success': 'mean',
            'severity_success': 'mean'
        }) * 100

        x_pos = np.arange(len(severity_success))
        width = 0.35

        ax1.bar(x_pos - width/2, severity_success['location_success'], width,
                label='Location Accuracy', color='skyblue', alpha=0.7)
        ax1.bar(x_pos + width/2, severity_success['severity_success'], width,
                label='Severity Classification', color='lightcoral', alpha=0.7)

        ax1.set_xlabel('Severity Class')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate by Severity Class', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(severity_success.index)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative error distribution
        for severity in ['Mild', 'Moderate', 'Severe']:
            mask = self.df['severity_class'] == severity
            if mask.any():
                errors = np.maximum.reduce([
                    self.df.loc[mask, 'notch_x_error_pct'],
                    self.df.loc[mask, 'notch_depth_error_pct'],
                    self.df.loc[mask, 'notch_width_error_pct']
                ])
                sorted_errors = np.sort(errors)
                cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                ax2.plot(sorted_errors, cumulative, marker='o', label=severity,
                        color=color_map[severity], linewidth=2)

        ax2.set_xlabel('Maximum Parameter Error (%)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Error Distribution by Severity', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Error magnitude by severity
        severity_errors = []
        for severity in ['Mild', 'Moderate', 'Severe']:
            mask = self.df['severity_class'] == severity
            if mask.any():
                max_errors = np.maximum.reduce([
                    self.df.loc[mask, 'notch_x_error_pct'],
                    self.df.loc[mask, 'notch_depth_error_pct'],
                    self.df.loc[mask, 'notch_width_error_pct']
                ])
                severity_errors.append(max_errors)

        ax3.boxplot(severity_errors, labels=['Mild', 'Moderate', 'Severe'])
        ax3.set_ylabel('Maximum Parameter Error (%)')
        ax3.set_title('Error Range by Severity Class', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Parameter-wise severity performance
        param_performance = []
        for param in ['notch_x', 'notch_depth', 'notch_width']:
            error_col = f'{param}_error_pct'
            for severity in ['Mild', 'Moderate', 'Severe']:
                mask = self.df['severity_class'] == severity
                if mask.any():
                    mean_error = self.df.loc[mask, error_col].mean()
                    param_performance.append({
                        'Parameter': param.replace('_', ' ').title(),
                        'Severity': severity,
                        'Mean Error (%)': mean_error
                    })

        perf_df = pd.DataFrame(param_performance)
        sns.barplot(data=perf_df, x='Parameter', y='Mean Error (%)',
                   hue='Severity', ax=ax4, palette=[color_map[s] for s in ['Mild', 'Moderate', 'Severe']])
        ax4.set_title('Mean Error by Parameter and Severity', fontweight='bold')
        ax4.legend(title='Severity')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for the analysis."""
        print("Generating summary statistics...")

        stats = []

        # Overall statistics
        for param in ['notch_x', 'notch_depth', 'notch_width']:
            error_col = f'{param}_error_pct'
            stats.append({
                'Parameter': param.replace('_', ' ').title(),
                'Mean Error (%)': self.df[error_col].mean(),
                'Median Error (%)': self.df[error_col].median(),
                'Std Error (%)': self.df[error_col].std(),
                'Max Error (%)': self.df[error_col].max(),
                'R²': np.corrcoef(self.df[f'{param}_true'], self.df[f'{param}_est'])[0, 1]**2
            })

        # Severity-based statistics
        severity_stats = self.df.groupby('severity_class').agg({
            'location_success': 'mean',
            'severity_success': 'mean',
            'test_id': 'count'
        }).round(3)

        return pd.DataFrame(stats), severity_stats

    def save_all_plots(self, output_dir: str = "Claude_res"):
        """Save all plots to the specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Saving plots to {output_path}/")

        # Generate and save plots
        fig1 = self.create_plot1_accuracy_scatter()
        fig1.savefig(output_path / "inverse_accuracy_scatter.png", dpi=300, bbox_inches='tight')
        fig1.savefig(output_path / "inverse_accuracy_scatter.pdf", bbox_inches='tight')

        fig2 = self.create_plot2_error_distribution()
        fig2.savefig(output_path / "inverse_error_distribution.png", dpi=300, bbox_inches='tight')
        fig2.savefig(output_path / "inverse_error_distribution.pdf", bbox_inches='tight')

        fig3 = self.create_plot3_severity_cumulative()
        fig3.savefig(output_path / "inverse_severity_cumulative.png", dpi=300, bbox_inches='tight')
        fig3.savefig(output_path / "inverse_severity_cumulative.pdf", bbox_inches='tight')

        # Generate and save summary statistics
        param_stats, severity_stats = self.generate_summary_statistics()

        param_stats.to_csv(output_path / "parameter_statistics.csv", index=False)
        severity_stats.to_csv(output_path / "severity_statistics.csv")

        # Save processed data
        self.df.to_csv(output_path / "processed_inverse_results.csv", index=False)

        print("All plots and statistics saved successfully!")

        # Print summary
        print("\n" + "="*50)
        print("INVERSE PROBLEM RESULTS SUMMARY")
        print("="*50)
        print(f"Total test cases: {len(self.df)}")
        print(f"Severity distribution: {self.df['severity_class'].value_counts().to_dict()}")
        print(f"Overall location success rate: {self.df['location_success'].mean():.1%}")
        print(f"Overall severity success rate: {self.df['severity_success'].mean():.1%}")
        print("\nParameter Statistics:")
        print(param_stats.to_string(index=False))
        print("\nSeverity Statistics:")
        print(severity_stats.to_string())

        plt.close('all')

def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = InverseResultsAnalyzer("/home/mecharoy/Thesis/Results/results/inverseresults.txt")

    # Save all plots
    analyzer.save_all_plots()

if __name__ == "__main__":
    main()