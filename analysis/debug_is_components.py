"""
Simple debug script to inspect IS components with breakpoints.
"""

import sys
from pathlib import Path
import argparse

from analysis.is_checkpoint1 import (
    extract_query_terms,
    coverage,
    redundancy,
    consistency,
    information_sufficiency,
    load_trajectories,
    compute_idf_cache
)

def main():
    parser = argparse.ArgumentParser(description='Debug IS components')
    parser.add_argument('--log_dir', type=str, 
                       default='deepresearch_logs/val/20251215_212118',
                       help='Directory with trajectory JSONL files')
    parser.add_argument('--output_dir', type=str,
                       default='deepresearch_outputs/val/20251215_212118',
                       help='Directory with result JSON files')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    
    if not log_dir.exists():
        print(f"Error: {log_dir} does not exist")
        return
    
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        return
    
    print(f"Loading trajectories from: {log_dir}")
    trajectories = load_trajectories(str(log_dir), str(output_dir))
    print(f"Loaded {len(trajectories)} trajectories")
    
    if not trajectories:
        print("No trajectories found!")
        return
    
    # Get first trajectory
    (qid, ridx), traj_data = list(trajectories.items())[0]
    question = traj_data['question']
    evidence_sequence = traj_data['evidence_sequence']
    print(f"\nSample Query (ID={qid}): {question[:100]}...")
    print(f"Rollout has {len(evidence_sequence)} turns")
    
    # Compute IDF cache
    print("\nComputing IDF cache...")
    idf_cache = compute_idf_cache(trajectories)
    print(f"IDF cache computed for {len(idf_cache)} terms")
    
    # Extract query terms
    query_terms = extract_query_terms(question)
    print(f"Query terms ({len(query_terms)}): {list(query_terms)[:15]}...")
    
    # Check a few turns
    print("\n" + "="*80)
    print("DEBUGGING IS COMPONENTS - Breakpoints will trigger at LLM calls")
    print("="*80)
    
    for turn in [0, 1, 2, 3, 4, 5]:
        if turn >= len(evidence_sequence):
            break
        
        K_t_text = evidence_sequence[turn]
        if not K_t_text:
            print(f"\n=== Turn {turn}: No evidence ===")
            continue
        
        K_t = [d.strip() for d in K_t_text.split('\n\n') if d.strip()]
        if not K_t:
            print(f"\n=== Turn {turn}: Empty evidence after parsing ===")
            continue
        
        print(f"\n{'='*80}")
        print(f"=== Turn {turn} ===")
        print(f"{'='*80}")
        print(f"  Evidence docs: {len(K_t)}")
        print(f"  Total evidence length: {sum(len(d) for d in K_t)} chars")
        print(f"  First doc preview: {K_t[0][:200]}...")
        print(f"\n  Computing IS components (breakpoints will trigger at LLM calls)...")
        
        # Compute components
        cov = coverage(K_t, query_terms, idf_cache)
        red = redundancy(K_t)
        cons = consistency(K_t, question, query_terms)
        non_red = 1.0 - red
        is_score = information_sufficiency(K_t, question, query_terms, idf_cache)
        
        print(f"\n  === Results ===")
        print(f"  Coverage: {cov:.6f}")
        print(f"  Consistency: {cons:.6f}")
        print(f"  Redundancy: {red:.6f}")
        print(f"  Non-Redundancy: {non_red:.6f}")
        print(f"  Final IS: {is_score:.6f}")
        
        # Identify bottleneck
        bottleneck = 'coverage' if is_score == cov else \
                    ('consistency' if is_score == cons else 'redundancy')
        print(f"  Bottleneck: {bottleneck}")
    
    print("\n" + "="*80)
    print("Debug session complete!")
    print("="*80)

if __name__ == '__main__':
    main()
