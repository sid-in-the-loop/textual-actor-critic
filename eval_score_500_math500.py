import argparse
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

from agent_system.multi_turn_rollout.mlmt_utils import (
    build_score_turn1_prompt,
    build_score_turn2_prompt
)

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

def run_score_eval():
    parser = argparse.ArgumentParser(description="ScoRe Checkpoint Evaluation on Math500 (Step 500 - LoRA Mode)")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model path")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to Step 500 actor/lora_adapter folder")
    parser.add_argument("--data_path", type=str, default="/home/ssmurali/mlmt/data/mlmt/math/test.parquet", help="Path to Math500 parquet")
    parser.add_argument("--output_path", type=str, default="results/score_step500_math500.jsonl", help="Output JSONL path")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    client = OpenAI(api_key=args.openai_api_key)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"🚀 Loading vLLM with Base: {args.base_model} and LoRA Adapter: {args.lora_path}")
    
    # Load base model and enable LoRA
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_loras=1,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=8192
    )
    lora_request = LoRARequest("score_adapter", 1, args.lora_path)
    
    # Parity Config
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=1024,
        n=1,
        stop=["<|eot_id|>", "<|end_of_text|>"]
    )

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    if args.num_samples:
        df = df.head(args.num_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    samples = []
    for i, row in df.iterrows():
        # Robustly extract question string
        question_raw = row['prompt']
        if isinstance(question_raw, (list, np.ndarray)) and len(question_raw) > 0:
            item = question_raw[0]
            if isinstance(item, dict) and 'content' in item:
                question = item['content']
            else:
                question = str(item)
        else:
            question = str(question_raw)
        
        # Robustly extract ground truth
        gt_raw = row['reward_model']['ground_truth'] if isinstance(row['reward_model'], dict) else row['reward_model']
        ground_truth = str(gt_raw)

        samples.append({
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "turns": {},
            "resolved_at_turn": None,
            "final_score": 0.0,
            "active": True
        })

    # --- Turn 1: Solve ---
    active_samples = [s for s in samples if s['active']]
    print(f"\n--- Turn 1: Solving (Active: {len(active_samples)}) ---")
    
    t1_prompts = []
    for s in active_samples:
        raw_p = build_score_turn1_prompt(s['question'])
        messages = [{"role": "user", "content": raw_p}]
        chat_p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t1_prompts.append(chat_p)

    outputs = llm.generate(t1_prompts, sampling_params, lora_request=lora_request)
    
    for s, out in zip(active_samples, outputs):
        s['turns']['1'] = {"content": out.outputs[0].text, "type": "actor_solve"}

    print(f"🔍 Evaluating Turn 1 solutions...")
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(evaluate_answer_soft, client, s['question'], s['turns']['1']['content'], s['ground_truth']): s 
            for s in active_samples
        }
        for future in tqdm(as_completed(futures), total=len(active_samples), desc="Scoring T1"):
            s = futures[future]
            is_correct, rationale = future.result()
            s['turns']['1']['correct'] = is_correct
            s['turns']['1']['rationale'] = rationale
            if is_correct:
                s['resolved_at_turn'] = 1
                s['final_score'] = 1.0
                s['active'] = False

    # --- Turn 2: Self-Correct (Only for unresolved) ---
    active_samples = [s for s in samples if s['active']]
    if active_samples:
        print(f"\n--- Turn 2: Self-Correcting (Active: {len(active_samples)}) ---")
        
        t2_prompts = []
        for s in active_samples:
            raw_p = build_score_turn2_prompt(s['question'], s['turns']['1']['content'])
            messages = [{"role": "user", "content": raw_p}]
            chat_p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            t2_prompts.append(chat_p)

    outputs = llm.generate(t2_prompts, sampling_params, lora_request=lora_request)
    
    for s, out in zip(active_samples, outputs):
        s['turns']['2'] = {"content": out.outputs[0].text, "type": "actor_correction"}

    print(f"🔍 Evaluating Turn 2 corrections...")
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(evaluate_answer_soft, client, s['question'], s['turns']['2']['content'], s['ground_truth']): s 
            for s in active_samples
        }
        for future in tqdm(as_completed(futures), total=len(active_samples), desc="Scoring T2"):
            s = futures[future]
            is_correct, rationale = future.result()
            s['turns']['2']['correct'] = is_correct
            s['turns']['2']['rationale'] = rationale
            if is_correct:
                s['resolved_at_turn'] = 2
                s['final_score'] = 1.0
                s['active'] = False
            else:
                print("\n✅ All samples resolved in Turn 1. Skipping Turn 2.")

    # Summary
    total = len(samples)
    t1_correct = sum(1 for s in samples if s['resolved_at_turn'] == 1)
    total_correct = sum(1 for s in samples if s['final_score'] > 0)
    
    print("\n" + "="*60)
    print(f"📊 ITERATIVE SCORE EVALUATION (STEP 500 - LORA MODE)")
    print("-" * 60)
    print(f"Total Samples:       {total}")
    print(f"Turn 1 Accuracy:     {t1_correct/total:>10.2%}")
    print(f"Cumulative Accuracy: {total_correct/total:>10.2%}")
    print(f"Self-Corrections:    {total_correct - t1_correct} cases (+{(total_correct - t1_correct)/total:.1%})")
    print("="*60)

    # Save results
    with open(args.output_path, "w") as f:
        for s in samples:
            if 'active' in s: s.pop('active')
            f.write(json.dumps(s) + "\n")
    
    print(f"\n✅ Results saved to {args.output_path}")

if __name__ == "__main__":
    run_score_eval()
