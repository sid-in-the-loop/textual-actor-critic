"""
Enhanced IS Analysis with Component Breakdowns

This script extends is_checkpoint1.py to provide detailed component analysis:
- Coverage scores
- Consistency scores  
- Redundancy scores
- Component-wise correlations with rewards
- Breakdown by turn

Usage:
    python analysis/is_checkpoint1_enhanced.py --log_dir <log_dir> --output_dir <output_dir>
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Import from base IS script
sys.path.insert(0, str(Path(__file__).parent))
from is_checkpoint1 import (
    load_all_trajectories, compute_metrics_aggregated,
    extract_query_terms
)


def analyze_components(aggregated_data: list) -> dict:
    """
    Analyze IS components and their relationships.
    
    Returns:
        Dictionary with component statistics and correlations
    """
    # Extract component scores
    coverage_vals = [d.get('coverage_mean', 0) for d in aggregated_data]
    consistency_vals = [d.get('consistency_mean', 0) for d in aggregated_data]
    redundancy_vals = [d.get('redundancy_mean', 0) for d in aggregated_data]
    non_redundancy_vals = [d.get('non_redundancy_mean', 0) for d in aggregated_data]
    is_vals = [d['is_mean'] for d in aggregated_data]
    rewards = [d['terminal_reward'] for d in aggregated_data]
    
    # Component statistics
    stats = {
        'coverage': {
            'mean': np.mean(coverage_vals),
            'median': np.median(coverage_vals),
            'std': np.std(coverage_vals),
            'min': np.min(coverage_vals),
            'max': np.max(coverage_vals)
        },
        'consistency': {
            'mean': np.mean(consistency_vals),
            'median': np.median(consistency_vals),
            'std': np.std(consistency_vals),
            'min': np.min(consistency_vals),
            'max': np.max(consistency_vals)
        },
        'redundancy': {
            'mean': np.mean(redundancy_vals),
            'median': np.median(redundancy_vals),
            'std': np.std(redundancy_vals),
            'min': np.min(redundancy_vals),
            'max': np.max(redundancy_vals)
        },
        'non_redundancy': {
            'mean': np.mean(non_redundancy_vals),
            'median': np.median(non_redundancy_vals),
            'std': np.std(non_redundancy_vals),
            'min': np.min(non_redundancy_vals),
            'max': np.max(non_redundancy_vals)
        },
        'is': {
            'mean': np.mean(is_vals),
            'median': np.median(is_vals),
            'std': np.std(is_vals),
            'min': np.min(is_vals),
            'max': np.max(is_vals)
        }
    }
    
    # Correlations with rewards (if rewards have variance)
    reward_variance = np.var(rewards)
    correlations = {}
    
    if reward_variance > 1e-10:
        correlations['is'] = spearmanr(is_vals, rewards)
        correlations['coverage'] = spearmanr(coverage_vals, rewards)
        correlations['consistency'] = spearmanr(consistency_vals, rewards)
        correlations['redundancy'] = spearmanr(redundancy_vals, rewards)
        correlations['non_redundancy'] = spearmanr(non_redundancy_vals, rewards)
    else:
        correlations = None
    
    # Which component is the bottleneck?
    bottleneck_analysis = []
    for d in aggregated_data:
        cov = d.get('coverage_mean', 0)
        cons = d.get('consistency_mean', 0)
        non_red = d.get('non_redundancy_mean', 0)
        is_score = d['is_mean']
        
        # Find which component is limiting IS
        if is_score == cov and cov <= cons and cov <= non_red:
            bottleneck = 'coverage'
        elif is_score == cons and cons <= cov and cons <= non_red:
            bottleneck = 'consistency'
        elif is_score == non_red and non_red <= cov and non_red <= cons:
            bottleneck = 'redundancy'
        else:
            bottleneck = 'unknown'
        
        bottleneck_analysis.append({
            'question_id': d['question_id'],
            'turn': d['turn'],
            'bottleneck': bottleneck,
            'is': is_score,
            'coverage': cov,
            'consistency': cons,
            'non_redundancy': non_red
        })
    
    bottleneck_counts = defaultdict(int)
    for b in bottleneck_analysis:
        bottleneck_counts[b['bottleneck']] += 1
    
    return {
        'stats': stats,
        'correlations': correlations,
        'bottleneck_analysis': bottleneck_analysis,
        'bottleneck_counts': dict(bottleneck_counts)
    }


def analyze_by_turn(aggregated_data: list) -> dict:
    """Analyze IS components by turn."""
    by_turn = defaultdict(lambda: {
        'is': [], 'coverage': [], 'consistency': [], 
        'redundancy': [], 'non_redundancy': [], 'rewards': []
    })
    
    for d in aggregated_data:
        turn = d['turn']
        by_turn[turn]['is'].append(d['is_mean'])
        by_turn[turn]['coverage'].append(d.get('coverage_mean', 0))
        by_turn[turn]['consistency'].append(d.get('consistency_mean', 0))
        by_turn[turn]['redundancy'].append(d.get('redundancy_mean', 0))
        by_turn[turn]['non_redundancy'].append(d.get('non_redundancy_mean', 0))
        by_turn[turn]['rewards'].append(d['terminal_reward'])
    
    turn_stats = {}
    for turn in sorted(by_turn.keys()):
        data = by_turn[turn]
        turn_stats[turn] = {
            'n_samples': len(data['is']),
            'is_mean': np.mean(data['is']),
            'is_std': np.std(data['is']),
            'coverage_mean': np.mean(data['coverage']),
            'consistency_mean': np.mean(data['consistency']),
            'redundancy_mean': np.mean(data['redundancy']),
            'non_redundancy_mean': np.mean(data['non_redundancy']),
            'reward_mean': np.mean(data['rewards']),
            'reward_std': np.std(data['rewards'])
        }
    
    return turn_stats


def generate_enhanced_report(metrics: dict, component_analysis: dict, turn_analysis: dict, output_dir: Path):
    """Generate enhanced report with component breakdowns."""
    report_path = output_dir / 'checkpoint1_enhanced_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Enhanced IS Analysis: Component Breakdown\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall IS statistics
        f.write("Overall IS Statistics:\n")
        f.write("-" * 80 + "\n")
        stats = component_analysis['stats']['is']
        f.write(f"  Mean: {stats['mean']:.6f}\n")
        f.write(f"  Median: {stats['median']:.6f}\n")
        f.write(f"  Std: {stats['std']:.6f}\n")
        f.write(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}\n\n")
        
        # Component statistics
        f.write("Component Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Component':<20} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-" * 80 + "\n")
        
        for comp_name in ['coverage', 'consistency', 'redundancy', 'non_redundancy']:
            comp_stats = component_analysis['stats'][comp_name]
            f.write(f"{comp_name:<20} {comp_stats['mean']:<12.6f} {comp_stats['median']:<12.6f} "
                   f"{comp_stats['std']:<12.6f} {comp_stats['min']:<12.6f} {comp_stats['max']:<12.6f}\n")
        
        f.write("\n")
        
        # Correlations
        if component_analysis['correlations']:
            f.write("Correlations with Terminal Rewards:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Component':<20} {'Spearman ρ':<15} {'p-value':<15}\n")
            f.write("-" * 80 + "\n")
            
            for comp_name, (rho, pval) in component_analysis['correlations'].items():
                f.write(f"{comp_name:<20} {rho:<15.6f} {pval:<15.6f}\n")
        else:
            f.write("Correlations: Cannot compute (all rewards are identical)\n")
        
        f.write("\n")
        
        # Bottleneck analysis
        f.write("Bottleneck Analysis (Which component limits IS?):\n")
        f.write("-" * 80 + "\n")
        for bottleneck, count in component_analysis['bottleneck_counts'].items():
            f.write(f"  {bottleneck}: {count} cases\n")
        
        f.write("\n")
        
        # Turn-by-turn analysis
        f.write("Turn-by-Turn Component Analysis:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Turn':<6} {'N':<6} {'IS Mean':<12} {'Coverage':<12} {'Consistency':<12} "
               f"{'Non-Red':<12} {'Reward Mean':<12}\n")
        f.write("-" * 80 + "\n")
        
        for turn in sorted(turn_analysis.keys()):
            t = turn_analysis[turn]
            f.write(f"{turn:<6} {t['n_samples']:<6} {t['is_mean']:<12.6f} {t['coverage_mean']:<12.6f} "
                   f"{t['consistency_mean']:<12.6f} {t['non_redundancy_mean']:<12.6f} "
                   f"{t['reward_mean']:<12.6f}\n")
        
        f.write("\n")
        
        # Original metrics
        f.write("Original Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Spearman ρ: {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.4f})\n")
        f.write(f"  ROC-AUC: {metrics['auc']:.4f}\n")
    
    print(f"\n✅ Enhanced report written to: {report_path}")


def plot_component_breakdown(component_analysis: dict, turn_analysis: dict, output_dir: Path):
    """Plot component breakdowns."""
    # Component distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stats = component_analysis['stats']
    
    # Plot 1: Component means comparison
    ax = axes[0, 0]
    components = ['coverage', 'consistency', 'non_redundancy', 'is']
    means = [stats[c]['mean'] for c in components]
    stds = [stats[c]['std'] for c in components]
    ax.bar(components, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_title('Component Score Means')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Turn-by-turn IS evolution
    ax = axes[0, 1]
    turns = sorted(turn_analysis.keys())
    is_means = [turn_analysis[t]['is_mean'] for t in turns]
    ax.plot(turns, is_means, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Turn')
    ax.set_ylabel('IS Mean')
    ax.set_title('IS Score Evolution by Turn')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Component evolution by turn
    ax = axes[1, 0]
    cov_means = [turn_analysis[t]['coverage_mean'] for t in turns]
    cons_means = [turn_analysis[t]['consistency_mean'] for t in turns]
    nonred_means = [turn_analysis[t]['non_redundancy_mean'] for t in turns]
    ax.plot(turns, cov_means, 'o-', label='Coverage', linewidth=2)
    ax.plot(turns, cons_means, 's-', label='Consistency', linewidth=2)
    ax.plot(turns, nonred_means, '^-', label='Non-Redundancy', linewidth=2)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Score')
    ax.set_title('Component Evolution by Turn')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Bottleneck distribution
    ax = axes[1, 1]
    bottleneck_counts = component_analysis['bottleneck_counts']
    if bottleneck_counts:
        bottlenecks = list(bottleneck_counts.keys())
        counts = list(bottleneck_counts.values())
        ax.bar(bottlenecks, counts, alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title('Bottleneck Distribution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'component_breakdown.png', dpi=150)
    plt.close()
    
    print(f"✅ Component breakdown plots saved to: {output_dir / 'component_breakdown.png'}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced IS analysis with component breakdowns')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing trajectory JSONL files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find corresponding output directory
    # Assuming log_dir is like deepresearch_logs/val/20251215_212118
    # and output_dir should be deepresearch_outputs/val/20251215_212118
    if 'deepresearch_logs' in str(log_dir):
        output_base = str(log_dir).replace('deepresearch_logs', 'deepresearch_outputs')
        result_dir = Path(output_base)
    else:
        # Try to find latest output directory
        val_dirs = sorted(Path('deepresearch_outputs/val').glob('*'))
        if val_dirs:
            result_dir = val_dirs[-1]
        else:
            result_dir = Path(args.output_dir)
    
    print(f"Loading trajectories from: {log_dir}")
    print(f"Loading results from: {result_dir}")
    
    # Load trajectories
    by_query = load_all_trajectories(str(log_dir), str(result_dir))
    
    # Compute metrics with components
    print("\nComputing IS scores with component breakdowns...")
    metrics = compute_metrics_aggregated(by_query, return_components=True)
    
    # Analyze components
    print("\nAnalyzing components...")
    component_analysis = analyze_components(metrics['aggregated_data'])
    
    # Analyze by turn
    print("\nAnalyzing by turn...")
    turn_analysis = analyze_by_turn(metrics['aggregated_data'])
    
    # Save aggregated data with components
    aggregated_file = output_dir / 'aggregated_data_with_components.json'
    with open(aggregated_file, 'w') as f:
        json.dump(metrics['aggregated_data'], f, indent=2)
    print(f"✅ Aggregated data saved to: {aggregated_file}")
    
    # Generate enhanced report
    generate_enhanced_report(metrics, component_analysis, turn_analysis, output_dir)
    
    # Plot component breakdowns
    plot_component_breakdown(component_analysis, turn_analysis, output_dir)
    
    print("\n✅ Enhanced analysis complete!")


if __name__ == '__main__':
    main()



