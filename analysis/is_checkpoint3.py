#!/usr/bin/env python3
"""
IS Checkpoint 2: Belief-Based Value Function (Refactored)

Clean implementation focused on:
- Per-turn belief trace logging to JSONL
- Evaluation metrics (Spearman, binning, AUC)
- No plotting (handle separately)
"""

from openai import OpenAI
import json
import numpy as np
import re
from collections import defaultdict
from scipy.stats import spearmanr
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import math

# =========================
# CONFIGURATION
# =========================
CMU_GATEWAY_BASE_URL = "https://ai-gateway.andrew.cmu.edu"
CMU_GATEWAY_API_KEY = "sk-dUplmEab2H7EFRaOISG1Ew"
LLM_MODEL = "gpt-4o-mini-2024-07-18"
USE_V2 = True          # switch: evaluate V2 vs V1
USE_MONOTONE = True    # prevents entailment flip from reducing value
BETA = 4.0             # margin gate sharpness (tune later: 1,2,4,8)

_client = OpenAI(
    api_key=CMU_GATEWAY_API_KEY,
    base_url=CMU_GATEWAY_BASE_URL,
    timeout=60
)

# =========================
# UTILITY
# =========================

def append_jsonl(path: str, record: dict):
    """Append a single JSON record to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# =========================
# HYPOTHESIS EXTRACTION & BELIEF COMPUTATION
# =========================

def extract_candidate_hypotheses_with_support(evidence_docs: List[str], question: str, max_candidates: int = 5) -> List[Dict]:
    """Extract candidate answer hypotheses with evidence support counts and snippets."""
    numbered_evidence = []
    for i, doc in enumerate(evidence_docs):
        numbered_evidence.append(f"[Doc {i+1}] {doc}")

    combined_evidence = "\n\n".join(numbered_evidence)

    prompt = f"""
You are extracting candidate answers for belief estimation, not answering the question.

Question:
{question}

Evidence documents (each document is independent):
{combined_evidence}

Your task:
Extract candidate answers that could directly fill the answer slot of the question.

Strict rules:
1. Only extract answers of the correct type for the question
   - If the question asks for a percentage, answers MUST be percentages
   - If it asks for a year/date, answers MUST be a year/date
   - If it asks for a name, answers MUST be a proper name
2. Do NOT extract vague statements, partial facts, or meta-information
3. If the evidence does not clearly contain an answer, return NO_ANSWER
5. Paraphrases of the same value count as the same hypothesis

For each hypothesis:
- hypothesis: the extracted answer (short, canonical form)
- count: number of distinct documents that explicitly support this hypothesis
- snippet: 1–2 sentences from ONE supporting document that explicitly answers the question with this value

Return at most {max_candidates} hypotheses.

If no hypothesis clearly answers the question, return:
[
  {{"hypothesis": "NO_ANSWER", "count": 0, "snippet": ""}}
]

Return ONLY valid JSON in this exact format:
[
  {{"hypothesis": "2015", "count": 3, "snippet": "The team was promoted in 2015..."}},
  {{"hypothesis": "1993", "count": 1, "snippet": "Restructuring occurred in 1993..."}}
]
"""

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0 
        )

        content = response.choices[0].message.content.strip()

        try:
            hypotheses = json.loads(content)
            if isinstance(hypotheses, list):
                result = []
                for h in hypotheses[:max_candidates]:
                    if isinstance(h, dict) and "hypothesis" in h:
                        hyp = h["hypothesis"].strip()
                        count = int(h.get("count", 0))
                        snippet = h.get("snippet", "").strip()
                        if hyp:
                            result.append({"hypothesis": hyp, "count": count, "snippet": snippet})
                if result:
                    return result
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON from text
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                hypotheses = json.loads(json_match.group(0))
                if isinstance(hypotheses, list):
                    result = []
                    for h in hypotheses[:max_candidates]:
                        if isinstance(h, dict) and "hypothesis" in h:
                            hyp = h["hypothesis"].strip()
                            count = int(h.get("count", 1))
                            snippet = h.get("snippet", "").strip()
                            if hyp:
                                result.append({"hypothesis": hyp, "count": count, "snippet": snippet})
                    if result:
                        return result
            except:
                pass

        return [{"hypothesis": "NO_ANSWER", "count": 0, "snippet": ""}]

    except Exception as e:
        print(f"Error extracting hypotheses: {e}")
        return [{"hypothesis": "NO_ANSWER", "count": 0, "snippet": ""}]

def normalize_hypothesis(hypothesis: str) -> str:
    """Normalize hypothesis for grouping (handle dates, numbers, etc.)."""
    hyp = hypothesis.lower().strip()

    # Normalize years (4-digit numbers)
    year_match = re.search(r'\b(19|20)\d{2}\b', hyp)
    if year_match:
        return year_match.group(0)

    # Normalize percentages
    pct_match = re.search(r'\b(\d+(?:\.\d+)?)%\b', hyp)
    if pct_match:
        return f"{pct_match.group(1)}%"

    # Normalize currency
    currency_match = re.search(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', hyp)
    if currency_match:
        return f"${currency_match.group(1)}"

    return hyp

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def build_normalized_belief(hypotheses_with_support: List[Dict]) -> Dict:
    """
    Aggregate support counts by normalized hypothesis and compute probabilities.

    Returns dict with:
      - items: list of {normalized, support_count, probability, hypothesis, snippet}
      - p1, p2, margin
      - top_normalized, top_hypothesis, top_snippet
    """
    agg_counts = defaultdict(int)
    agg_hypothesis = {}
    agg_snippet = {}

    # Filter NO_ANSWER early
    valid = [h for h in hypotheses_with_support if h.get("hypothesis", "").strip() != "NO_ANSWER"]

    if not valid:
        return {
            "items": [],
            "p1": 0.0,
            "p2": 0.0,
            "margin": 0.0,
            "top_normalized": "NO_ANSWER",
            "top_hypothesis": "NO_ANSWER",
            "top_snippet": ""
        }

    for h in valid:
        raw_h = (h.get("hypothesis") or "").strip()
        cnt = int(h.get("count", 0))
        snip = (h.get("snippet") or "").strip()
        norm = normalize_hypothesis(raw_h)

        agg_counts[norm] += max(0, cnt)

        # store a representative raw hypothesis (first seen)
        if norm not in agg_hypothesis and raw_h:
            agg_hypothesis[norm] = raw_h
        # store a representative snippet (first non-empty)
        if norm not in agg_snippet and snip:
            agg_snippet[norm] = snip

    total = sum(agg_counts.values())
    if total <= 0:
        return {
            "items": [],
            "p1": 0.0,
            "p2": 0.0,
            "margin": 0.0,
            "top_normalized": "NO_ANSWER",
            "top_hypothesis": "NO_ANSWER",
            "top_snippet": ""
        }

    items = []
    for norm, cnt in agg_counts.items():
        p = cnt / total
        items.append({
            "hypothesis": agg_hypothesis.get(norm, norm),
            "normalized": norm,
            "support_count": cnt,
            "probability": p,
            "snippet": agg_snippet.get(norm, "")
        })

    items.sort(key=lambda x: x["probability"], reverse=True)

    top = items[0]
    p1 = float(top["probability"])
    p2 = float(items[1]["probability"]) if len(items) > 1 else 0.0
    margin = max(0.0, p1 - p2)

    return {
        "items": items,
        "p1": p1,
        "p2": p2,
        "margin": margin,
        "top_normalized": top["normalized"],
        "top_hypothesis": top["hypothesis"],
        "top_snippet": top["snippet"]
    }

def compute_belief_concentration(hypotheses_with_support: List[Dict]) -> Tuple[float, str]:
    """
    Compute belief concentration using weighted support counts.
    
    Returns: (concentration: float, top_hypothesis: str)
    """
    if not hypotheses_with_support:
        return 0.0, "NO_ANSWER"

    valid_hypotheses = [h for h in hypotheses_with_support if h.get("hypothesis", "") != "NO_ANSWER"]
    if not valid_hypotheses:
        return 0.0, "NO_ANSWER"

    # Normalize and aggregate by hypothesis
    normalized_counts = defaultdict(int)
    hypothesis_snippets = {}

    for h in valid_hypotheses:
        hyp = h["hypothesis"]
        count = h.get("count", 1)
        snippet = h.get("snippet", "")

        normalized_hyp = normalize_hypothesis(hyp)
        normalized_counts[normalized_hyp] += count

        if normalized_hyp not in hypothesis_snippets:
            hypothesis_snippets[normalized_hyp] = snippet

    if not normalized_counts:
        return 0.0, "NO_ANSWER"

    # Compute probabilities from counts
    total_counts = sum(normalized_counts.values())
    probabilities = {h: count / total_counts for h, count in normalized_counts.items()}

    # Concentration = max probability
    max_prob = max(probabilities.values())
    top_hypothesis = max(probabilities.items(), key=lambda x: x[1])[0]

    # Optional: combine with gap between top and second
    sorted_probs = sorted(probabilities.values(), reverse=True)
    if len(sorted_probs) > 1:
        gap = sorted_probs[0] - sorted_probs[1]
        concentration = max_prob * (1.0 + gap) / 2.0
    else:
        concentration = max_prob

    return min(1.0, concentration), top_hypothesis

# Cache for entailment checks
_entailment_cache = {}

def check_entailment_support(question: str, hypothesis: str, supporting_snippet: str) -> float:
    """Check if the supporting snippet entails the hypothesis as the answer."""
    if hypothesis == "NO_ANSWER" or not supporting_snippet:
        return 0.0

    cache_key = f"{question}|||{hypothesis}|||{supporting_snippet[:100]}"
    if cache_key in _entailment_cache:
        return _entailment_cache[cache_key]

    prompt = f"""
You are verifying answer support, not reasoning or guessing.

Question:
{question}

Hypothesis (candidate answer):
{hypothesis}

Supporting Snippet:
{supporting_snippet}

Task:
Decide whether the supporting snippet EXPLICITLY states that the hypothesis is the answer to the question.

Strict criteria:
- The snippet must clearly and directly answer the question using the hypothesis value.
- The hypothesis value must appear explicitly (or as a clear numeric/date equivalent).
- The snippet must not merely mention the hypothesis or imply it.
- If the question asks for a number, date, percentage, or name, the snippet MUST contain that exact value.

Scoring:
- 1.0 → The snippet MUST answer/contain the answer to the question with the hypothesis.
- 0.0 → The snippet does NOT AT ALL answer the question with the hypothesis.

Return ONLY one number: 1.0 or 0.0
"""

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        content = response.choices[0].message.content.strip()
        number_match = re.search(r'(\d+(?:\.\d+)?)', content)
        if number_match:
            score = float(number_match.group(1))
            score = max(0.0, min(1.0, score))
            _entailment_cache[cache_key] = score
            return score

    except Exception as e:
        print(f"Error checking entailment: {e}")

    _entailment_cache[cache_key] = 0.0
    return 0.0

def belief_based_value_function(
    evidence_docs: List[str],
    question: str,
    previous_belief: Optional[float] = None,
    alpha: float = 0.3
) -> Tuple[float, Dict]:
    """
    Computes either:
      V1_raw = concentration * entailment
      V2_raw = (p1 * entailment) * sigmoid(BETA * margin)

    Returns chosen (possibly smoothed) value + metadata containing both.
    """
    if not evidence_docs:
        return 0.0, {
            "concentration": 0.0,
            "support": 0.0,
            "top_hypothesis": "NO_ANSWER",
            "snippet": "",
            "hypotheses": [],
            "p1": 0.0, "p2": 0.0, "margin": 0.0, "gate": 0.0,
            "raw_value_v1": 0.0,
            "raw_value_v2": 0.0
        }

    # 1) Extract candidates
    hypotheses_with_support = extract_candidate_hypotheses_with_support(evidence_docs, question)

    # 2) Proper normalized belief distribution
    belief = build_normalized_belief(hypotheses_with_support)
    p1, p2, margin = belief["p1"], belief["p2"], belief["margin"]
    top_hypothesis = belief["top_normalized"]  # use normalized as canonical top id
    supporting_snippet = belief["top_snippet"]

    # 3) Keep your old "concentration" behavior, but make it consistent with p1/p2
    #    (matches your previous: max_prob * (1+gap)/2)
    if p1 > 0 and (p2 > 0):
        concentration = min(1.0, p1 * (1.0 + (p1 - p2)) / 2.0)
    else:
        concentration = min(1.0, p1)

    # 4) Entailment check (use p1 threshold instead of concentration threshold)
    if top_hypothesis != "NO_ANSWER" and supporting_snippet and p1 >= 0.3:
        support = check_entailment_support(question, top_hypothesis, supporting_snippet)
    else:
        support = 0.0

    raw_v1 = concentration * support

    gate = sigmoid(BETA * margin)
    raw_v2 = (p1 * support) * gate

    # 5) Choose which raw score to use for the returned value
    raw_value = raw_v2 if USE_V2 else raw_v1

    # 6) Optional monotone clamp to prevent entailment flips from decreasing value
    if previous_belief is not None and USE_MONOTONE:
        raw_value = max(previous_belief, raw_value)

    # 7) Smoothing
    if previous_belief is not None:
        value = alpha * previous_belief + (1.0 - alpha) * raw_value
    else:
        value = raw_value

    metadata = {
        "concentration": concentration,
        "support": support,
        "top_hypothesis": top_hypothesis,       # normalized id for top
        "snippet": supporting_snippet,

        # normalized belief distribution (already aggregated)
        "hypotheses": belief["items"],

        # V2 ingredients
        "p1": p1,
        "p2": p2,
        "margin": margin,
        "gate": gate,

        # both raws for offline comparison
        "raw_value_v1": raw_v1,
        "raw_value_v2": raw_v2
    }

    return value, metadata

# =========================
# EVALUATION
# =========================

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

    # Spearman
    if np.var(rewards) == 0:
        print("\nSpearman ρ       : UNDEFINED (no reward variance)")
        rho, pval = np.nan, np.nan
    else:
        rho, pval = spearmanr(is_scores, rewards)
        print("\nSpearman rank correlation")
        print("-" * 32)
        print(f"ρ (rho)          : {rho:.4f}")
        print(f"p-value          : {pval:.4e}")

    # Binned Analysis
    print("\nBinned Analysis (IS deciles)")
    print("-" * 32)

    if len(is_scores) >= 10:
        sorted_indices = np.argsort(is_scores)
        sorted_is = is_scores[sorted_indices]
        sorted_rewards = rewards[sorted_indices]

        n_bins = 10
        bin_size = len(sorted_is) // n_bins

        print("Bin  | IS Range     | Avg Reward | Samples")
        print("-" * 45)

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_is)

            bin_is = sorted_is[start_idx:end_idx]
            bin_rewards = sorted_rewards[start_idx:end_idx]

            avg_reward = np.mean(bin_rewards)
            p_reward_1 = np.mean(bin_rewards > 0)
            print(f"{i:3d}  | [{bin_is.min():.3f}, {bin_is.max():.3f}] | {avg_reward:.4f} ({p_reward_1:.2%}) | {len(bin_is):3d}")

    # ROC-AUC
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
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(binary_rewards, is_scores)
            print("\nROC-AUC")
            print("-" * 32)
            print(f"AUC               : {auc:.4f}")
        except ImportError:
            auc = np.nan
            print("\nROC-AUC           : SKIPPED (sklearn not available)")

    # Interpretation
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

# =========================
# LOADING & MAIN EVALUATION
# =========================

def load_trajectories(log_dir: str, output_dir: str, sample_size=299):
    """Load trajectories from JSONL files and results from JSON files."""
    from pathlib import Path
    import random

    log_path = Path(log_dir)
    output_path = Path(output_dir)

    trajectories = {}
    traj_files = sorted(log_path.glob('trajectory_*.jsonl'))

    print(f"Found {len(traj_files)} trajectory files")

    # Sample trajectory files if requested
    if sample_size and len(traj_files) > sample_size:
        print(f"Sampling {sample_size} trajectory files from {len(traj_files)} total...")
        traj_files = random.sample(traj_files, sample_size)

    for traj_file in tqdm(traj_files, desc="Loading trajectories"):
        match = re.match(r'trajectory_(\d+)_(\d+)\.jsonl', traj_file.name)
        if not match:
            continue

        question_id = int(match.group(1))
        rollout_idx = int(match.group(2))

        evidence_sequence = []
        question = None

        with open(traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                turn_data = json.loads(line)

                if question is None:
                    question = turn_data.get('question', '')

                evidence_text = turn_data.get('K_t_combined', '')
                if evidence_text:
                    evidence_sequence.append(evidence_text)

        result_file = output_path / f"result_{question_id}_{rollout_idx}.json"
        terminal_reward = 0

        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    terminal_reward = result_data.get('reward_sum', 0)
            except:
                pass

        trajectories[(question_id, rollout_idx)] = {
            'question': question,
            'evidence_sequence': evidence_sequence,
            'terminal_reward': terminal_reward
        }

    return trajectories

def evaluate_directory(log_dir: str, output_dir: str):
    """Evaluate belief-based IS correlation and log per-turn belief traces."""
    from pathlib import Path

    print("=" * 72)
    print("IS Checkpoint 2: Belief-Based Value Function (Refactored)")
    print("=" * 72)
    print(f"Log directory    : {log_dir}")
    print(f"Output directory : {output_dir}")
    print()

    # Setup belief trace log
    belief_log_path = f"{output_dir}/belief_trace_2.jsonl"
    open(belief_log_path, "w").close()  # Reset file

    # Load trajectories
    print("Loading trajectories...")
    trajectories = load_trajectories(log_dir, output_dir, sample_size=299)
    print(f"Loaded {len(trajectories)} trajectories (sampled)")

    if not trajectories:
        print("Error: No trajectories found!")
        return

    print("\nComputing belief-based IS scores and logging belief traces...")

    is_scores = []
    terminal_rewards = []

    for (qid, ridx), traj_data in tqdm(trajectories.items(), desc="Processing trajectories"):
        question = traj_data['question']
        evidence_sequence = traj_data['evidence_sequence']
        terminal_reward = int(traj_data['terminal_reward'])

        previous_belief = None

        for turn_idx, K_t_text in enumerate(evidence_sequence):
            if not K_t_text:
                continue

            K_t = [d.strip() for d in K_t_text.split('\n\n') if d.strip()]
            if not K_t:
                continue

            value, metadata = belief_based_value_function(
                K_t,
                question,
                previous_belief=previous_belief
            )

            # Build hypothesis distribution - FIXED: Use support_count not count
            hypotheses = metadata.get("hypotheses", [])
            total_support = sum(h.get("support_count", 0) for h in hypotheses) or 1

            hypotheses_log = []
            for h in hypotheses:
                hyp = h.get("hypothesis", "")
                norm = h.get("normalized", normalize_hypothesis(hyp))
                cnt = h.get("support_count", 0)
                prob = h.get("probability", 0.0)
                hypotheses_log.append({
                    "hypothesis": hyp,
                    "normalized": norm,
                    "support_count": cnt,
                    "probability": prob
                })

            # FIXED: Build record OUTSIDE the hypothesis loop
            record = {
                "question_id": qid,
                "rollout_idx": ridx,
                "turn_idx": turn_idx,
                "question": question,
                "num_docs": len(K_t),
                "terminal_reward": terminal_reward,

                "hypotheses": hypotheses_log,

                "top_hypothesis": metadata["top_hypothesis"],  # now normalized
                "top_hypothesis_normalized": metadata["top_hypothesis"],

                "concentration": metadata["concentration"],
                "supporting_snippet": metadata["snippet"],
                "entailment_score": metadata["support"],

                "p1": metadata.get("p1", 0.0),
                "p2": metadata.get("p2", 0.0),
                "margin": metadata.get("margin", 0.0),
                "gate": metadata.get("gate", 0.0),

                "raw_value_v1": metadata.get("raw_value_v1", 0.0),
                "raw_value_v2": metadata.get("raw_value_v2", 0.0),

                "previous_value": previous_belief,
                "smoothed_value": value
            }

            append_jsonl(belief_log_path, record)
            previous_belief = value

        # Final turn IS (for correlation only)
        if evidence_sequence:
            K_t_text = evidence_sequence[-1]
            if K_t_text:
                K_t = [d.strip() for d in K_t_text.split('\n\n') if d.strip()]
                if K_t:
                    belief_value, _ = belief_based_value_function(K_t, question, previous_belief=None)
                    is_scores.append(belief_value)
                    terminal_rewards.append(terminal_reward)

    print(f"Logged belief traces to: {belief_log_path}")
    print(f"Computed {len(is_scores)} final-turn IS scores")

    # Evaluate correlation
    metrics = evaluate_is_with_binning(is_scores, terminal_rewards, name="Belief-Based IS")

    return metrics

def main():
    """Main entry point."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Evaluate belief-based IS correlation with terminal rewards'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        required=True,
        help='Directory containing trajectory JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory containing result JSON files'
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

    evaluate_directory(str(log_dir), str(output_dir))

if __name__ == '__main__':
    main()