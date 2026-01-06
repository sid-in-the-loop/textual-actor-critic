import argparse
import json
import os
import ast
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

from agent_system.multi_turn_rollout.mlmt_utils import (
    prepare_mlmt_turn1_prompt,
    prepare_mlmt_feedback_prompt,
    prepare_mlmt_refinement_prompt
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

def build_benchmark_prompt(tokenizer, problem: str):
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following math problem step by step. "
                "Then give the final answer.\n\n"
                f"Problem:\n{problem}"
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

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

def extract_question(prompt_col):
    if isinstance(prompt_col, (list, pd.Series)) and len(prompt_col) > 0:
        if isinstance(prompt_col[0], dict) and 'content' in prompt_col[0]:
            return prompt_col[0]['content']
    return str(prompt_col)

def run_mlmt_iter_inference():
    # Force stable V0 engine for multi-model reliability
    os.environ["VLLM_USE_V1"] = "0"
    
    parser = argparse.ArgumentParser(description="MLMT Iterative Inference (Separate High/Low Actors)")
    parser.add_argument("--low_model", type=str, required=True, help="Low-Level Actor model path (HF)")
    parser.add_argument("--high_model", type=str, required=True, help="High-Level Guider model path (HF)")
    parser.add_argument("--data_path", type=str, default="/home/ssmurali/mlmt/data/mlmt/math/test.parquet", help="Path to Math500 parquet")
    parser.add_argument("--input_path", type=str, default=None, help="Path to previous results to resume from")
    parser.add_argument("--turns", type=int, default=3, help="Max total number of turns (T = 2K+1)")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for refinement turns")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35, help="GPU memory for EACH engine (total = 2 * this)")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key for semantic judge")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers for OpenAI judge")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (per engine)")
    args = parser.parse_args()

    client = OpenAI(api_key=args.openai_api_key)

    if args.output_path is None:
        args.output_path = f"results/mlmt_math500_T{args.turns}_iterative.jsonl"

    print(f"🚀 Initializing TWO vLLM Engines in bfloat16 (Memory Partition: {args.gpu_memory_utilization} each)")
    
    # Engine 1: Low-Level Actor
    print(f"📦 Loading Low-Level Actor: {args.low_model}")
    low_llm = LLM(
        model=args.low_model, 
        trust_remote_code=True, 
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        enforce_eager=True # Avoid graph capture overhead for multi-model
    )
    
    # Engine 2: High-Level Actor (Guider)
    print(f"📦 Loading High-Level Actor: {args.high_model}")
    high_llm = LLM(
        model=args.high_model, 
        trust_remote_code=True, 
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        enforce_eager=True # Avoid graph capture overhead for multi-model
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.low_model)
    
    # Sampling params matched with high-accuracy config
    params_turn1 = SamplingParams(temperature=0.0, max_tokens=2048, n=1)
    params_turn_feedback = SamplingParams(temperature=args.temperature, max_tokens=512, n=1)
    params_turn_refine = SamplingParams(temperature=args.temperature, max_tokens=2048, n=1)

    # Initialize samples container
    samples = []
    if args.input_path:
        print(f"📖 Resuming from input: {args.input_path}")
        # (Resume logic identical to iter_inference_math500.py)
        is_jsonl = args.input_path.endswith('.jsonl')
        data_rows = []
        if not is_jsonl:
            try:
                df_in = pd.read_csv(args.input_path)
                data_rows = df_in.to_dict('records')
            except:
                is_jsonl = True
        if is_jsonl:
            with open(args.input_path, 'r') as f:
                for line in f:
                    if line.strip(): data_rows.append(json.loads(line))

        for i, row in enumerate(data_rows):
            if "turns" in row and "1" in row["turns"]:
                q, gt = row.get('question', ''), row.get('ground_truth', '')
                t1_content = row["turns"]["1"]["content"]
            else:
                q = str(row.get('question', row.get('prompt', '')))
                gt = str(row.get('ground_truth', ''))
                t1_content = str(row.get('turn1_output', row.get('model_output', '')))
            
            samples.append({
                "index": i, "question": q, "ground_truth": gt,
                "turns": {"1": {"content": t1_content, "type": "actor_solve"}},
                "resolved_at_turn": None, "final_score": 0.0, "active": True
            })
    else:
        print(f"Loading data: {args.data_path}")
        df = pd.read_parquet(args.data_path)
        if args.num_samples: df = df.head(args.num_samples)
        for i, row in df.iterrows():
            samples.append({
                "index": i, "question": extract_question(row['prompt']),
                "ground_truth": row['reward_model']['ground_truth'],
                "turns": {}, "resolved_at_turn": None, "final_score": 0.0, "active": True
            })

    # Main iterative loop
    for t in range(1, args.turns + 1):
        active_samples = [s for s in samples if s['active']]
        if not active_samples: break

        is_actor_turn = (t % 2 == 1)
        role = "Low-Level Actor" if is_actor_turn else "High-Level Guider"
        engine = low_llm if is_actor_turn else high_llm
        print(f"\n--- Turn {t} ({role}) | Active: {len(active_samples)} ---")

        if t == 1 and args.input_path:
            responses = [s['turns']['1']['content'] for s in active_samples]
        else:
            prompts = []
            for s in active_samples:
                if t == 1:
                    prompts.append(build_benchmark_prompt(tokenizer, s['question']))
                elif not is_actor_turn:
                    prompts.append(prepare_mlmt_feedback_prompt(s['question'], s['turns'][str(t-1)]['content']))
                else:
                    prompts.append(prepare_mlmt_refinement_prompt(s['question'], s['turns'][str(t-2)]['content'], s['turns'][str(t-1)]['content']))

            if t == 1: current_params = params_turn1
            elif not is_actor_turn: current_params = params_turn_feedback
            else: current_params = params_turn_refine

            outputs = engine.generate(prompts, current_params)
            responses = [out.outputs[0].text for out in outputs]

            for s, resp in zip(active_samples, responses):
                s['turns'][str(t)] = {"content": resp, "type": "actor_solve" if is_actor_turn else "guider_feedback"}

        if is_actor_turn:
            print(f"🔍 Evaluating Turn {t} solutions...")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_sample = {executor.submit(evaluate_answer_soft, client, s['question'], s['turns'][str(t)]['content'], s['ground_truth']): s for s in active_samples}
                for future in tqdm(as_completed(future_to_sample), total=len(active_samples), desc="Scoring"):
                    sample = future_to_sample[future]
                    is_correct, rationale = future.result()
                    sample['turns'][str(t)]['score'] = 1.0 if is_correct else 0.0
                    if is_correct:
                        sample['resolved_at_turn'] = t
                        sample['final_score'] = 1.0
                        sample['active'] = False

    # Final Summary
    total = len(samples)
    t1_acc = sum(1 for s in samples if s['resolved_at_turn'] == 1) / total
    final_acc = sum(1 for s in samples if s['final_score'] > 0) / total
    print(f"\n📊 SUMMARY\nTurn 1 Accuracy: {t1_acc:.2%}\nFinal Accuracy:  {final_acc:.2%}")

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    with open(args.output_path, "w") as f:
        for s in samples:
            if 'active' in s: s.pop('active')
            f.write(json.dumps(s) + "\n")
    print(f"✅ Saved to {args.output_path}")

if __name__ == "__main__":
    run_mlmt_iter_inference()

