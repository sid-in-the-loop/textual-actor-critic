import json
import os
import argparse
import ast
import pandas as pd
from typing import Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

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
"judgement": "correct" or "incorrect",
"confidence": "high" or "medium" or "low"
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
            import time
            time.sleep(2 ** attempt)
    return False, "Unknown Error"

def process_sample_sync(client, i, row, is_csv=False):
    """Worker function to process one sample."""
    if is_csv:
        # 1. Extract Question
        if 'question' in row:
            q = str(row['question'])
        elif 'prompt' in row:
            try:
                prompt_list = ast.literal_eval(row['prompt'])
                q = prompt_list[0]['content'] if isinstance(prompt_list, list) and len(prompt_list) > 0 else str(row['prompt'])
            except:
                q = str(row['prompt'])
        else:
            q = "Unknown Question"

        # 2. Extract Ground Truth
        if 'ground_truth' in row:
            gt = str(row['ground_truth'])
        elif 'reward_model' in row:
            try:
                rm_dict = ast.literal_eval(row['reward_model'])
                gt = rm_dict.get('ground_truth', '')
            except:
                gt = str(row['reward_model'])
        else:
            gt = ""

        # 3. Detect Mode (Dynamics vs single-turn)
        if 'turn1_output' in row and 'turn3_output' in row:
            # Multi-turn CSV
            t1_correct, t1_rat = evaluate_answer_soft(client, q, str(row['turn1_output']), gt)
            t3_correct, t3_rat = evaluate_answer_soft(client, q, str(row['turn3_output']), gt)
            
            # Use original scores if available for discrepancy check
            orig_t1 = float(row.get('turn1_score', 0)) > 0
            orig_t3 = float(row.get('turn3_score', 0)) > 0
            has_disc = (t1_correct != orig_t1) or (t3_correct != orig_t3)
            
            status = "STABLE"
            note = ""
            if not t1_correct and t3_correct:
                status = "IMPROVED (0->1)"
                note = f"T3 Rationale: {t3_rat}"
            elif t1_correct and not t3_correct:
                status = "DEGRADED (1->0)"
                note = f"T3 Rationale: {t3_rat}"
            elif not t1_correct and not t3_correct:
                status = "FAILED (0->0)"
            else:
                status = "STABLE (1->1)"

            return {
                "index": i,
                "t1_correct": t1_correct,
                "t3_correct": t3_correct,
                "has_disc": has_disc,
                "status": status,
                "note": note,
                "is_csv_multi": True
            }
        else:
            # Single-turn CSV
            pred = str(row.get('model_output', row.get('turn1_output', '')))
            correct, rationale = evaluate_answer_soft(client, q, pred, gt)
            return {
                "index": i,
                "correct": correct,
                "rationale": rationale,
                "is_csv_single": True
            }
    else:
        # JSONL format (multi-turn)
        q = row.get('question', '')
        gt = row.get('ground_truth', '')
        
        # Turn 1 is always expected
        t1_data = row.get('turns', {}).get('1', {})
        t1_correct, t1_rat = evaluate_answer_soft(client, q, t1_data.get('content', ''), gt)
        orig_t1 = float(t1_data.get('score', 0)) > 0
        
        # Turn 3 might be missing if sample resolved at Turn 1
        t3_data = row.get('turns', {}).get('3', {})
        if t3_data:
            t3_correct, t3_rat = evaluate_answer_soft(client, q, t3_data.get('content', ''), gt)
            orig_t3 = float(t3_data.get('score', 0)) > 0
        else:
            # Fallback to T1 if T3 doesn't exist (e.g. resolved at T1)
            t3_correct, t3_rat = t1_correct, t1_rat
            orig_t3 = orig_t1
        has_disc = (t1_correct != orig_t1) or (t3_correct != orig_t3)
        
        status = "STABLE"
        note = ""
        if not t1_correct and t3_correct:
            status = "IMPROVED (0->1)"
            note = f"T3 Rationale: {t3_rat}"
        elif t1_correct and not t3_correct:
            status = "DEGRADED (1->0)"
            note = f"T3 Rationale: {t3_rat}"
        elif not t1_correct and not t3_correct:
            status = "FAILED (0->0)"
        else:
            status = "STABLE (1->1)"

        return {
            "index": i,
            "t1_correct": t1_correct,
            "t3_correct": t3_correct,
            "has_disc": has_disc,
            "status": status,
            "note": note,
            "is_jsonl": True
        }

def run_reevaluation():
    parser = argparse.ArgumentParser(description="Parallel re-evaluation of Math500 dynamics.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to JSONL log or CSV results")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI Key")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers")
    args = parser.parse_args()

    client = OpenAI(api_key=args.openai_api_key)

    print(f"📖 Reading {args.input_path}...")
    is_csv = args.input_path.endswith('.csv')
    
    try:
        if is_csv:
            df = pd.read_csv(args.input_path)
            data = df.to_dict('records')
        else:
            with open(args.input_path, "r") as f:
                data = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"🚀 Re-evaluating {len(data)} samples using {args.workers} workers...")
    
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {executor.submit(process_sample_sync, client, i, row, is_csv=is_csv): i for i, row in enumerate(data)}
        
        for future in tqdm(as_completed(future_to_idx), total=len(data), desc="Processing"):
            r = future.result()
            results.append(r)
            
            # Print logic
            if r.get('is_csv_single'):
                if r['index'] % 50 == 0 or not r['correct']:
                    tqdm.write(f"{r['index']:<5} | {str(r['correct']):<7} | {r['rationale'][:80]}...")
            else:
                if r['has_disc'] or "IMPROVED" in r['status'] or "DEGRADED" in r['status'] or r['index'] % 50 == 0:
                    disc_str = "YES" if r['has_disc'] else "no"
                    tqdm.write(f"{r['index']:<5} | {int(r['t1_correct']):<3} | {int(r['t3_correct']):<3} | {r['status']:<18} | {disc_str:<5} | {r['note']}")

    total = len(data)
    if total == 0: return
    results.sort(key=lambda x: x['index'])

    print("\n" + "="*60)
    print(f"📊 FINAL SUMMARY: {os.path.basename(args.input_path)}")
    print("-" * 60)
    
    if results[0].get('is_csv_single'):
        accuracy = sum(1 for r in results if r['correct']) / total
        print(f"Verified Accuracy: {accuracy:>10.2%}")
        print(f"Total Correct:     {sum(1 for r in results if r['correct']):>4} / {total}")
    else:
        print(f"Verified Turn 1 Accuracy: {sum(1 for r in results if r['t1_correct'])/total:>10.2%}")
        print(f"Verified Turn 3 Accuracy: {sum(1 for r in results if r['t3_correct'])/total:>10.2%}")
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
    run_reevaluation()
