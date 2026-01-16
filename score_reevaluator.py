import json
import os
import argparse
import pandas as pd
from typing import Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import time

# --- Math-specific evaluation prompt ---
SOFT_EVALUATION_PROMPT = """
Please evaluate if the final mathematical answer in the Predicted Answer is equivalent to the Ground Truth.

**EVALUATION CRITERIA:**
- The answer is CORRECT if it is mathematically equivalent to the ground truth, even if the format is different.
- **Flexibility Allowed:**
    - Fractions vs. Decimals (e.g., 1/2 vs 0.5)
    - LaTeX vs. Plain text (e.g., \\frac{{1}}{{2}} vs 1/2 or \\pi vs pi)
    - Simplified vs. Unsimplified forms (e.g., \\sqrt{{8}} vs 2\\sqrt{{2}})
    - Inclusion or exclusion of units (e.g., 5 cm vs 5) unless the unit is the core of the question.
    - Presence of "The answer is:" or LaTeX boxing \\boxed{{...}}
- **Strictness Required:**
    - The numerical value or algebraic expression must be identical in meaning.
    - Contradictory values or incorrect signs must be marked INCORRECT.

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {predicted_answer}

Please respond with a JSON object:
{{
"rationale": "briefly explain the mathematical comparison",
"judgement": "correct" or "incorrect"
}}
"""

def evaluate_answer_soft(client: OpenAI, question: str, predicted_answer: str, ground_truth: str) -> Tuple[bool, str]:
    """Synchronous soft evaluation with retries."""
    prompt = SOFT_EVALUATION_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        predicted_answer=predicted_answer
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={ "type": "json_object" },
                timeout=30.0
            )
            res = json.loads(response.choices[0].message.content)
            is_correct = res.get('judgement', 'incorrect').lower() == 'correct'
            return is_correct, res.get('rationale', 'No rationale provided')
        except Exception:
            if attempt == max_retries - 1:
                return False, "Failed after retries"
            time.sleep(2 ** attempt)
    return False, "Unknown Error"

def process_sample_score(client, i, row):
    """Worker function to process one ScoRe sample."""
    q = row.get('question', '')
    gt = row.get('ground_truth', '')
    
    # Handle both potential formats (flat turn1/turn2 or turns dict)
    turns = row.get('turns', {})
    if not turns:
        # Fallback to flat format if needed
        t1_content = row.get('turn1', {}).get('content', '')
        t2_content = row.get('turn2', {}).get('content', '')
        orig_t1 = bool(row.get('turn1', {}).get('correct', False))
        orig_t2 = bool(row.get('turn2', {}).get('correct', False))
    else:
        t1_content = turns.get('1', {}).get('content', '')
        t2_content = turns.get('2', {}).get('content', '')
        orig_t1 = bool(turns.get('1', {}).get('correct', False))
        orig_t2 = bool(turns.get('2', {}).get('correct', False))

    # Evaluate Turn 1
    t1_correct, t1_rat = evaluate_answer_soft(client, q, t1_content, gt)
    
    # Evaluate Turn 2 (if present)
    if not t2_content.strip():
        # If Turn 2 is missing, it means it was resolved at Turn 1 or skipped.
        # Consider it stable based on Turn 1 outcome.
        t2_correct = t1_correct
        t2_rat = "Turn 2 skipped (Resolved at T1)"
    else:
        t2_correct, t2_rat = evaluate_answer_soft(client, q, t2_content, gt)
    
    # Check for discrepancies
    has_disc = (t1_correct != orig_t1) or (t2_correct != orig_t2)
    
    status = "STABLE"
    note = ""
    if not t1_correct and t2_correct:
        status = "IMPROVED (0->1)"
        note = f"T2 Rationale: {t2_rat}"
    elif t1_correct and not t2_correct:
        status = "DEGRADED (1->0)"
        note = f"T2 Rationale: {t2_rat}"
    elif not t1_correct and not t2_correct:
        status = "FAILED (0->0)"
    else:
        status = "STABLE (1->1)"

    return {
        "index": i,
        "t1_correct": t1_correct,
        "t2_correct": t2_correct,
        "has_disc": has_disc,
        "status": status,
        "note": note
    }

def run_score_reevaluation():
    parser = argparse.ArgumentParser(description="Parallel re-evaluation of ScoRe JSONL results.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to ScoRe JSONL results")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI Key")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers")
    args = parser.parse_args()

    client = OpenAI(api_key=args.openai_api_key)

    print(f"📖 Reading {args.input_path}...")
    try:
        with open(args.input_path, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"🚀 Re-evaluating {len(data)} samples using {args.workers} workers...")
    
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {executor.submit(process_sample_score, client, i, row): i for i, row in enumerate(data)}
        
        for future in tqdm(as_completed(future_to_idx), total=len(data), desc="Processing"):
            r = future.result()
            results.append(r)
            
            if r['has_disc'] or "IMPROVED" in r['status'] or "DEGRADED" in r['status'] or r['index'] % 50 == 0:
                disc_str = "DISC!" if r['has_disc'] else "ok"
                tqdm.write(f"{r['index']:<5} | T1: {int(r['t1_correct'])} | T2: {int(r['t2_correct'])} | {r['status']:<18} | {disc_str:<5} | {r['note'][:100]}")

    total = len(data)
    if total == 0: return
    results.sort(key=lambda x: x['index'])

    print("\n" + "="*60)
    print(f"📊 ScoRe RE-EVALUATION SUMMARY: {os.path.basename(args.input_path)}")
    print("-" * 60)
    
    t1_acc = sum(1 for r in results if r['t1_correct']) / total
    t2_acc = sum(1 for r in results if (r['t1_correct'] or r['t2_correct'])) / total
    
    print(f"Verified Turn 1 Accuracy: {t1_acc:>10.2%}")
    print(f"Verified Cumulative Acc:  {t2_acc:>10.2%}")
    print("-" * 60)
    improved = sum(1 for r in results if r['status'].startswith("IMPROVED"))
    degraded = sum(1 for r in results if r['status'].startswith("DEGRADED"))
    stable = sum(1 for r in results if r['status'].startswith("STABLE (1->1)"))
    failed = sum(1 for r in results if r['status'] == "FAILED (0->0)")
    
    print(f"Flipped 0 -> 1 (Improvement): {improved:>4} ({improved/total:>6.1%})")
    print(f"Flipped 1 -> 0 (Degradation): {degraded:>4} ({degraded/total:>6.1%})")
    print(f"Stable Correct (1 -> 1):     {stable:>4} ({stable/total:>6.1%})")
    print(f"Persistent Fail (0 -> 0):    {failed:>4} ({failed/total:>6.1%})")
    print("-" * 60)
    discrepancies = sum(1 for r in results if r['has_disc'])
    print(f"Total Discrepancies vs original scores: {discrepancies}")
    print("="*60)

if __name__ == "__main__":
    run_score_reevaluation()

