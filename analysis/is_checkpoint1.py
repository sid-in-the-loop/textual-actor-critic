from openai import OpenAI
import hashlib
import json
import numpy as np
import re
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
# =========================
# LLM CONFIG
# =========================
CMU_GATEWAY_BASE_URL = "https://ai-gateway.andrew.cmu.edu"
CMU_GATEWAY_API_KEY = "sk-dUplmEab2H7EFRaOISG1Ew"
LLM_MODEL = "gpt-4o-mini-2024-07-18"

_client = OpenAI(
    api_key=CMU_GATEWAY_API_KEY,
    base_url=CMU_GATEWAY_BASE_URL,
    timeout=60
)

# =========================
# QUERY TERM EXTRACTION
# =========================
def extract_query_terms(query: str) -> set[str]:
    words = re.findall(r"\b\w+\b", query.lower())
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
        'where', 'when', 'why', 'how', 'if', 'then', 'else', 'because',
        'while', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
        'get', 'make', 'take', 'give', 'put', 'set', 'go', 'come', 'see', 'know',
        'think', 'say', 'tell', 'ask', 'try', 'use', 'find', 'work', 'call',
        'want', 'need', 'let', 'help', 'show', 'move', 'play', 'run', 'turn',
        'start', 'stop', 'keep', 'leave', 'bring', 'begin', 'seem', 'feel',
        'become', 'look', 'mean', 'live', 'believe', 'hold', 'write', 'provide',
        'include', 'continue', 'allow', 'follow', 'change', 'lead', 'understand'
    }
    return {w for w in words if w not in stopwords and len(w) > 2}

# =========================
# COVERAGE
# =========================
TAU = 1.0

def coverage(K_t: list[str], query_terms: set[str], idf_cache: dict[str, float]) -> float:
    if not K_t or not query_terms:
        return 0.0

    idf_weights = {t: idf_cache.get(t, 0.0) for t in query_terms}
    total_idf = sum(idf_weights.values())
    if total_idf == 0:
        return 0.0

    score = 0.0
    for term in query_terms:
        df = sum(1 for doc in K_t if term in doc.lower())
        presence = 1.0 - np.exp(-df / TAU)
        score += (idf_weights[term] / total_idf) * presence

    return min(1.0, score)

# =========================
# REDUNDANCY
# =========================
def redundancy(K_t: list[str], shingle_size: int = 3) -> float:
    if len(K_t) <= 1:
        return 0.0

    def shingles(text: str) -> set[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return {
            " ".join(words[i:i + shingle_size])
            for i in range(len(words) - shingle_size + 1)
        }

    sets = [shingles(doc) for doc in K_t]
    max_sims = []

    for i, s1 in enumerate(sets):
        sims = []
        for j, s2 in enumerate(sets):
            if i == j:
                continue
            inter = len(s1 & s2)
            union = len(s1 | s2)
            sims.append(inter / union if union else 0.0)
        max_sims.append(max(sims) if sims else 0.0)

    return float(np.mean(max_sims))

# =========================
# LLM ANSWER EXTRACTION
# =========================
def llm_extract_answer(question: str, evidence: list[str]) -> str:
    """
    Extract a single explicit factual claim from evidence that is relevant
    to answering the question. Partial claims are allowed.

    This function is a hypothesis probe, NOT a final answer generator.
    """

    evidence_text = "\n\n".join(evidence).strip()

    prompt = f"""
You are extracting factual claims from evidence.

Question:
{question}

Evidence:
{evidence_text}

Instructions:
- Extract ONE explicitly stated factual claim from the evidence.
- The claim must be relevant to answering the question.
- Partial facts (dates, events, restructurings, promotions) are VALID.
- Do NOT infer, guess, or combine facts.
- Do NOT require the claim to fully answer the question.
- If no explicit relevant fact is stated, return exactly: NO_ANSWER
- One sentence only.

Return format:
<ANSWER>...</ANSWER>
"""

    # resp = _client.chat.completions.create(
    #     model=LLM_MODEL,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.0
    # )

    # content = resp.choices[0].message.content.strip()

    # match = re.search(r"<ANSWER>(.*?)</ANSWER>", content, re.DOTALL)
    # if not match:
    #     raise RuntimeError(f"Invalid LLM extraction format:\n{content}")

    # answer = match.group(1).strip()
    # # Normalize trivial refusals
    # if answer.upper() == "NO_ANSWER":
    #     return "NO_ANSWER"
    return evidence_text
# =========================
# LLM AGREEMENT
# =========================
_AGREEMENT_CACHE = {}

def llm_agreement(question: str, answers: list[str]) -> float:
    if len(answers) < 2:
        return 0.0

    key = hashlib.md5(
        json.dumps({"q": question, "a": sorted(answers)}).encode()
    ).hexdigest()

    if key in _AGREEMENT_CACHE:
        return _AGREEMENT_CACHE[key]

    answers_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))

    prompt  = f"""
You are judging agreement between factual hypotheses.

Question:
{question}

Extracted factual claims:
{answers_text}

Instructions:
- Group answers that refer to the same underlying factual claim.
- Identify the dominant claim (the one supported by most answers).
- Return a number between 0 and 1 equal to:

    (# answers supporting dominant claim) / (total answers)

- Minor outliers should NOT cause disagreement.
- Do NOT use outside knowledge.
- Do NOT require full correctness.

Return ONLY the number.
"""

    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    content = resp.choices[0].message.content.strip()
    
    # Extract number from response (handle various formats)
    numbers = re.findall(r"\d+\.?\d*", content)
    score = float(numbers[0])
    score = max(0.0, min(1.0, score))

    _AGREEMENT_CACHE[key] = score
    return score

# =========================
# CONSISTENCY
# =========================
def consistency(
    K_t: list[str],
    question: str,
    query_terms: set[str] = None,
    n_bootstrap: int = 6
) -> float:
    if len(K_t) <= 1:
        return 1.0

    rng = np.random.RandomState(42)
    answers = []

    for _ in tqdm(range(n_bootstrap), desc="Consistency bootstrap", leave=False):
        idx = rng.choice(len(K_t), size=max(1, int(len(K_t))), replace=True)
        subset = [K_t[i] for i in idx]
        ans = llm_extract_answer(question, subset)
        if ans != "NO_ANSWER":
            answers.append(ans)

    answer_rate = len(answers) / n_bootstrap
    if answer_rate == 0.0:
        return 0.0

    agreement = llm_agreement(question, answers)
    return answer_rate * agreement

# =========================
# INFORMATION SUFFICIENCY
# =========================
def information_sufficiency(
    K_t: list[str],
    question: str,
    query_terms: set[str],
    idf_cache: dict[str, float]
) -> float:
    cov = coverage(K_t, query_terms, idf_cache)
    cons = consistency(K_t, question, query_terms)
    non_red = 1.0 - redundancy(K_t)

    return min(cov, cons, non_red)

def evaluate_is_with_binning(is_scores, terminal_rewards, name="IS"):
    """Enhanced evaluation with binning and detailed analysis."""
    is_scores = np.array(is_scores)
    rewards = np.array(terminal_rewards)

    print("\n" + "=" * 72)
    print(f"{name} — Enhanced Predictive Quality Evaluation")
    print("=" * 72)

    print(f"Samples          : {len(is_scores)}")
    print(f"IS range         : [{is_scores.min():.4f}, {is_scores.max():.4f}]")
    print(f"Reward range     : [{rewards.min():.4f}, {rewards.max():.4f}]")
    print(f"Reward variance  : {np.var(rewards):.6f}")

    # ---------- Spearman ----------
    if np.var(rewards) == 0:
        print("\nSpearman ρ       : UNDEFINED (no reward variance)")
        rho, pval = np.nan, np.nan
    else:
        rho, pval = spearmanr(is_scores, rewards)
        print("\nSpearman rank correlation")
        print("-" * 32)
        print(f"ρ (rho)          : {rho:.4f}")
        print(f"p-value          : {pval:.4e}")

    # ---------- Binned Analysis ----------
    print("\nBinned Analysis (IS deciles)")
    print("-" * 32)

    if len(is_scores) >= 10:
        # Sort by IS score
        sorted_indices = np.argsort(is_scores)
        sorted_is = is_scores[sorted_indices]
        sorted_rewards = rewards[sorted_indices]

        # Create deciles
        n_bins = 10
        bin_size = len(sorted_is) // n_bins

        print("Bin  | IS Range     | Avg Reward | Samples")
        print("-" * 45)

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_is)

            bin_is = sorted_is[start_idx:end_idx]
            bin_rewards = sorted_rewards[start_idx:end_idx]

            print(f"{i:3d}  | [{bin_is.min():.3f}, {bin_is.max():.3f}] | {bin_rewards.mean():.4f}     | {len(bin_is):3d}")

    # ---------- ROC-AUC ----------
    if len(np.unique(rewards)) > 2:
        threshold = np.median(rewards)
        binary_rewards = (rewards > threshold).astype(int)
        print("\nROC-AUC setup")
        print("-" * 32)
        print(f"Binary threshold : median = {threshold:.4f}")
    else:
        binary_rewards = rewards.astype(int)

    if len(np.unique(binary_rewards)) < 2:
        auc = np.nan
        print("\nROC-AUC           : UNDEFINED (single class)")
    else:
        auc = roc_auc_score(binary_rewards, is_scores)
        print("\nROC-AUC")
        print("-" * 32)
        print(f"AUC               : {auc:.4f}")

    # ---------- Verdict ----------
    print("\nInterpretation")
    print("-" * 32)

    if not np.isnan(rho):
        print("Monotonic signal :", "✓" if rho > 0 else "✗")
        if abs(rho) > 0.3:
            print("Strength         :", "Strong" if abs(rho) > 0.5 else "Moderate")
        else:
            print("Strength         :", "Weak")
    else:
        print("Signal assessment: INSUFFICIENT DATA")

    return {
        'samples': len(is_scores),
        'is_range': [float(is_scores.min()), float(is_scores.max())],
        'reward_range': [float(rewards.min()), float(rewards.max())],
        'spearman_rho': float(rho) if not np.isnan(rho) else None,
        'spearman_pval': float(pval) if not np.isnan(pval) else None,
        'auc': float(auc) if not np.isnan(auc) else None
    }

def plot_coverage_consistency_over_turns(trajectories: Dict, idf_cache: Dict, output_dir: str):
    """Plot coverage and consistency progression over turns, grouped by terminal reward."""
    print("\nGenerating plots...")

    # Group trajectories by terminal reward and turns
    reward_groups = defaultdict(lambda: defaultdict(list))

    for (qid, ridx), traj_data in tqdm(trajectories.items(), desc="Grouping trajectories for plotting"):
        question = traj_data['question']
        evidence_sequence = traj_data['evidence_sequence']
        terminal_reward = int(traj_data['terminal_reward'])
        turns = len(evidence_sequence)

        if turns == 0:
            continue

        query_terms = extract_query_terms(question)

        # Compute IS components for each turn
        coverage_scores = []
        consistency_scores = []
        redundancy_scores = []

        for turn in range(turns):
            K_t_text = evidence_sequence[turn]
            if not K_t_text:
                continue

            K_t = [d.strip() for d in K_t_text.split('\n\n') if d.strip()]
            if not K_t:
                continue

            cov = coverage(K_t, query_terms, idf_cache)
            cons = consistency(K_t, question)
            red = redundancy(K_t)

            coverage_scores.append(cov)
            consistency_scores.append(cons)
            redundancy_scores.append(red)

        if coverage_scores:  # Only if we have data
            reward_groups[terminal_reward][turns].append({
                'coverage': coverage_scores,
                'consistency': consistency_scores,
                'redundancy': redundancy_scores
            })

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('IS Components Over Turns (Grouped by Terminal Reward)', fontsize=16)

    colors = ['red', 'blue']  # 0: red, 1: blue
    reward_labels = ['Terminal Reward = 0', 'Terminal Reward = 1']

    for reward_val in [0, 1]:
        if reward_val not in reward_groups:
            continue

        color = colors[reward_val]
        label = reward_labels[reward_val]

        # Collect all trajectories for this reward group
        all_coverage = defaultdict(list)
        all_consistency = defaultdict(list)

        for turns, traj_list in reward_groups[reward_val].items():
            for traj in traj_list:
                for turn_idx, (cov, cons) in enumerate(zip(traj['coverage'], traj['consistency'])):
                    all_coverage[turn_idx].append(cov)
                    all_consistency[turn_idx].append(cons)

        # Plot coverage
        max_turns = max(all_coverage.keys()) if all_coverage else 0
        turns_range = list(range(max_turns + 1))

        coverage_means = []
        coverage_stds = []
        consistency_means = []
        consistency_stds = []

        for turn in turns_range:
            if turn in all_coverage:
                coverage_means.append(np.mean(all_coverage[turn]))
                coverage_stds.append(np.std(all_coverage[turn]))
            else:
                coverage_means.append(0)
                coverage_stds.append(0)

            if turn in all_consistency:
                consistency_means.append(np.mean(all_consistency[turn]))
                consistency_stds.append(np.std(all_consistency[turn]))
            else:
                consistency_means.append(0)
                consistency_stds.append(0)

        # Coverage plot
        axes[0, reward_val].errorbar(turns_range, coverage_means, yerr=coverage_stds,
                                   color=color, marker='o', label=label, capsize=3)
        axes[0, reward_val].set_title(f'Coverage - {label}')
        axes[0, reward_val].set_xlabel('Turn')
        axes[0, reward_val].set_ylabel('Coverage Score')
        axes[0, reward_val].grid(True, alpha=0.3)
        axes[0, reward_val].set_ylim(0, 1)

        # Consistency plot
        axes[1, reward_val].errorbar(turns_range, consistency_means, yerr=consistency_stds,
                                   color=color, marker='s', label=label, capsize=3)
        axes[1, reward_val].set_title(f'Consistency - {label}')
        axes[1, reward_val].set_xlabel('Turn')
        axes[1, reward_val].set_ylabel('Consistency Score')
        axes[1, reward_val].grid(True, alpha=0.3)
        axes[1, reward_val].set_ylim(0, 1)

    plt.tight_layout()
    plot_path = f"{output_dir}/is_components_over_turns.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()

def analyze_by_reward_and_turns(trajectories: Dict, idf_cache: Dict):
    """Analyze IS components grouped by terminal reward and number of turns."""
    print("\n" + "=" * 72)
    print("IS Analysis by Terminal Reward and Turns")
    print("=" * 72)

    # Group by terminal reward and turns
    reward_turn_groups = defaultdict(lambda: defaultdict(list))

    for (qid, ridx), traj_data in tqdm(trajectories.items(), desc="Analyzing by reward and turns"):
        question = traj_data['question']
        evidence_sequence = traj_data['evidence_sequence']
        terminal_reward = int(traj_data['terminal_reward'])
        turns = len(evidence_sequence)

        if turns == 0 or not evidence_sequence[-1]:
            continue

        # Use final turn
        K_t_text = evidence_sequence[-1]
        K_t = [d.strip() for d in K_t_text.split('\n\n') if d.strip()]

        if not K_t:
            continue

        query_terms = extract_query_terms(question)

        # Compute IS components
        cov = coverage(K_t, query_terms, idf_cache)
        cons = consistency(K_t, question)
        red = redundancy(K_t)
        is_score = information_sufficiency(K_t, question, query_terms, idf_cache)

        reward_turn_groups[terminal_reward][turns].append({
            'coverage': cov,
            'consistency': cons,
            'redundancy': red,
            'is_score': is_score
        })

    # Print summary
    for reward_val in [0, 1]:
        if reward_val not in reward_turn_groups:
            continue

        print(f"\nTerminal Reward = {reward_val}")
        print("-" * 40)

        for turns in sorted(reward_turn_groups[reward_val].keys()):
            trajs = reward_turn_groups[reward_val][turns]

            cov_mean = np.mean([t['coverage'] for t in trajs])
            cons_mean = np.mean([t['consistency'] for t in trajs])
            red_mean = np.mean([t['redundancy'] for t in trajs])
            is_mean = np.mean([t['is_score'] for t in trajs])

            print(f"{turns} turns | Cov: {cov_mean:.3f} | Cons: {cons_mean:.3f} | Red: {red_mean:.3f} | IS: {is_mean:.3f} | n={len(trajs)}")

def evaluate_is(is_scores, terminal_rewards, name="IS"):
    """Backward compatibility wrapper."""
    return evaluate_is_with_binning(is_scores, terminal_rewards, name)

# =========================
# TRAJECTORY LOADING & EVALUATION
# =========================
def load_trajectories(log_dir: str, output_dir: str):
    """Load trajectories from JSONL files and results from JSON files."""
    from pathlib import Path
    import json
    
    log_path = Path(log_dir)
    output_path = Path(output_dir)
    
    trajectories = {}
    
    # Find all trajectory JSONL files
    traj_files = sorted(log_path.glob('trajectory_*.jsonl'))
    
    print(f"Found {len(traj_files)} trajectory files")
    
    for traj_file in tqdm(traj_files, desc="Loading trajectories"):
        # Parse filename: trajectory_{question_id}_{rollout_idx}.jsonl
        match = re.match(r'trajectory_(\d+)_(\d+)\.jsonl', traj_file.name)
        if not match:
            continue
        
        question_id = int(match.group(1))
        rollout_idx = int(match.group(2))
        
        # Load trajectory JSONL
        evidence_sequence = []
        question = None
        
        with open(traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                turn_data = json.loads(line)
                
                if question is None:
                    question = turn_data.get('question', '')
                
                # Extract K_t (list of evidence strings)
                K_t = turn_data.get('K_t', [])
                if isinstance(K_t, list):
                    K_t_combined = '\n\n'.join(K_t)
                else:
                    K_t_combined = turn_data.get('K_t_combined', '')
                
                evidence_sequence.append(K_t_combined)
        
        if not evidence_sequence or not question:
            continue
        
        # Load result JSON
        result_file = output_path / f'result_{question_id}_{rollout_idx}.json'
        if not result_file.exists():
            continue
        
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        terminal_reward = result.get('reward_sum', 0.0)
        
        trajectories[(question_id, rollout_idx)] = {
            'question': question,
            'evidence_sequence': evidence_sequence,
            'terminal_reward': terminal_reward
        }
    
    return trajectories

def compute_idf_cache(trajectories: dict) -> dict[str, float]:
    """Compute IDF cache from all evidence across all trajectories."""
    print("Collecting all documents for IDF computation...")
    
    # Collect all evidence documents (flattened)
    all_docs_flat = []
    for traj_data in tqdm(trajectories.values(), desc="Collecting documents"):
        for K_t_text in traj_data['evidence_sequence']:
            if K_t_text:
                docs = [d.strip() for d in K_t_text.split('\n\n') if d.strip()]
                all_docs_flat.extend(docs)
    
    print(f"Collected {len(all_docs_flat)} total documents")
    
    # Collect all query terms
    print("Extracting query terms...")
    all_terms = set()
    for traj_data in tqdm(trajectories.values(), desc="Extracting terms"):
        terms = extract_query_terms(traj_data['question'])
        all_terms.update(terms)
    
    print(f"Found {len(all_terms)} unique query terms")
    
    # Compute IDF for each term (optimized: check against flat list)
    N = len(all_docs_flat)
    idf_cache = {}
    
    # Pre-lowercase all docs for faster lookup
    print("Computing IDF scores...")
    all_docs_lower = [doc.lower() for doc in tqdm(all_docs_flat, desc="Lowercasing docs")]
    
    for term in tqdm(all_terms, desc="Computing IDF"):
        doc_freq = sum(1 for doc in all_docs_lower if term in doc)
        if doc_freq == 0:
            idf_cache[term] = 0.0
        else:
            idf_cache[term] = np.log(N / doc_freq) if N > 0 else 0.0
    
    return idf_cache

def evaluate_directory(log_dir: str, output_dir: str, generate_plots: bool = False):
    """Evaluate IS correlation for trajectories in given directories."""
    from pathlib import Path
    
    print("=" * 72)
    print("IS Correlation Evaluation")
    print("=" * 72)
    print(f"Log directory    : {log_dir}")
    print(f"Output directory : {output_dir}")
    print()
    
    # Load trajectories
    print("Loading trajectories...")
    trajectories = load_trajectories(log_dir, output_dir)
    print(f"Loaded {len(trajectories)} trajectories")
    
    if not trajectories:
        print("Error: No trajectories found!")
        return
    
    # Compute IDF cache
    print("\nComputing IDF cache...")
    idf_cache = compute_idf_cache(trajectories)
    print(f"IDF cache computed for {len(idf_cache)} terms")
    
    # Compute IS scores for final turn of each trajectory
    print("\nComputing IS scores...")
    is_scores = []
    terminal_rewards = []
    
    for (qid, ridx), traj_data in tqdm(trajectories.items(), desc="Processing trajectories"):
        question = traj_data['question']
        evidence_sequence = traj_data['evidence_sequence']
        terminal_reward = traj_data['terminal_reward']
        
        if not evidence_sequence:
            continue
        
        # Use final turn's evidence
        K_t_text = evidence_sequence[-1]
        if not K_t_text:
            continue
        # Split into documents
        K_t = [d.strip() for d in K_t_text.split('\n\n') if d.strip()]
        if not K_t:
            continue
        
        # Extract query terms
        query_terms = extract_query_terms(question)
        
        # Compute IS
        is_score = information_sufficiency(K_t, question, query_terms, idf_cache)
        
        is_scores.append(is_score)
        terminal_rewards.append(terminal_reward)
    
    print(f"Computed {len(is_scores)} IS scores")

    # Evaluate correlation with enhanced binning
    metrics = evaluate_is(is_scores, terminal_rewards, name="IS")

    # Generate plots if requested
    if generate_plots:
        try:
            plot_coverage_consistency_over_turns(trajectories, idf_cache, output_dir)
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")

    # Analyze by reward and turns
    analyze_by_reward_and_turns(trajectories, idf_cache)

    return metrics

def main():
    """Main entry point for IS correlation evaluation."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Evaluate IS correlation with terminal rewards'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        required=True,
        help='Directory containing trajectory JSONL files (e.g., deepresearch_logs/val/20251215_212118)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory containing result JSON files (e.g., deepresearch_outputs/val/20251215_212118)'
    )
    parser.add_argument(
        '--generate_plots',
        action='store_true',
        help='Generate plots for IS components over turns'
    )
    
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)

    if not log_dir.exists():
        print(f"Error: Log directory does not exist: {log_dir}")
        return

    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        return

    evaluate_directory(str(log_dir), str(output_dir), args.generate_plots)

if __name__ == '__main__':
    main()