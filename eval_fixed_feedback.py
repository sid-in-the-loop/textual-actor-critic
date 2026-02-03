import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

from agent_system.multi_turn_rollout.mlmt_utils import (
    prepare_mlmt_refinement_prompt
)

# Math-specific evaluation prompt
SOFT_EVALUATION_PROMPT = """
Please evaluate if the final mathematical answer in the Predicted Answer is equivalent to the Ground Truth.

**EVALUATION CRITERIA:**
- The answer is CORRECT if it is mathematically equivalent to the ground truth, even if the format is different.
- **Flexibility Allowed:** Fractions/Decimals, LaTeX/Plain text, Simplified/Unsimplified.
- **Strictness Required:** Numerical value or algebraic expression must be identical in meaning.

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {predicted_answer}

Please respond with a JSON object:
{{
"rationale": "briefly explain the mathematical comparison",
"judgement": "correct" or "incorrect"
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

def evaluate_answer_soft(client: OpenAI, question: str, predicted_answer: str, ground_truth: str) -> bool:
    prompt = SOFT_EVALUATION_PROMPT.format(question=question, ground_truth=ground_truth, predicted_answer=predicted_answer)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={ "type": "json_object" },
            timeout=30.0
        )
        res = json.loads(response.choices[0].message.content)
        return res.get('judgement', 'incorrect').lower() == 'correct'
    except Exception:
        return False

def extract_question(prompt_col):
    if isinstance(prompt_col, (list, pd.Series)) and len(prompt_col) > 0:
        if isinstance(prompt_col[0], dict) and 'content' in prompt_col[0]:
            return prompt_col[0]['content']
    return str(prompt_col)

def run_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, 
                        help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--data_path", type=str, default="/home/ssmurali/mlmt/data/mlmt/math/test.parquet")
    parser.add_argument("--output_path", type=str, default="results/eval_fixed_feedback_base.jsonl")
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    client = OpenAI(api_key=args.openai_api_key)
    
    # Optimized LLM init: max_model_len=4096 speeds up graph capture significantly
    if args.adapter_path:
        print(f"🚀 Loading vLLM with model: {args.model_path}")
        print(f"📎 Loading LoRA adapter from: {args.adapter_path}")
        llm = LLM(
            model=args.model_path, 
            tensor_parallel_size=args.tp, 
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            enable_lora=True
        )
        # Create LoRA request object
        lora_request = LoRARequest("math_adapter", 1, args.adapter_path)
    else:
        print(f"🚀 Loading vLLM with model: {args.model_path}")
        llm = LLM(
            model=args.model_path, 
            tensor_parallel_size=args.tp, 
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True
        )
        lora_request = None
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    df = pd.read_parquet(args.data_path)
    questions = [extract_question(row['prompt']) for _, row in df.iterrows()]
    ground_truths = [row['reward_model']['ground_truth'] for _, row in df.iterrows()]
    
    # 1. Turn 1: Solve
    print("🚀 Running Turn 1 (Initial Solve)...")
    params_t1 = SamplingParams(temperature=0.7, max_tokens=2048)
    prompts_t1 = [build_benchmark_prompt(tokenizer, q) for q in questions]
    outputs_t1 = llm.generate(prompts_t1, params_t1, lora_request=lora_request)
    y0_list = [out.outputs[0].text for out in outputs_t1]

    # 2. Evaluate Turn 1
    print("🔍 Evaluating Turn 1 solutions (100 workers)...")
    t1_results = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(evaluate_answer_soft, client, q, y0, gt): i 
                   for i, (q, y0, gt) in enumerate(zip(questions, y0_list, ground_truths))}
        
        results_map = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring T1"):
            idx = futures[future]
            results_map[idx] = future.result()
        
        t1_results = [results_map[i] for i in range(len(questions))]

    t1_acc = sum(t1_results) / len(t1_results)
    print(f"📊 Turn 1 Accuracy: {t1_acc:.2%}")

    # 3. Filter Incorrect for Turn 3
    incorrect_indices = [i for i, correct in enumerate(t1_results) if not correct]
    print(f"ℹ️ {len(incorrect_indices)} samples were incorrect. Proceeding to Turn 3...")

    y1_list = [None] * len(questions)
    t3_results = [False] * len(questions)

    if incorrect_indices:
        # 4. Turn 3: Refine
        print(f"🚀 Running Turn 3 (Fixed Feedback)...")
        fixed_feedback = "Score feedback"
        params_t3 = SamplingParams(temperature=0.7, max_tokens=2048)
        prompts_t3 = [prepare_mlmt_refinement_prompt(questions[i], y0_list[i], fixed_feedback) 
                      for i in incorrect_indices]
        
        outputs_t3 = llm.generate(prompts_t3, params_t3, lora_request=lora_request)
        
        for idx, out in zip(incorrect_indices, outputs_t3):
            y1_list[idx] = out.outputs[0].text

        # 5. Evaluate Turn 3
        print("🔍 Evaluating Turn 3 solutions (100 workers)...")
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = {executor.submit(evaluate_answer_soft, client, questions[i], y1_list[i], ground_truths[i]): i 
                       for i in incorrect_indices}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring T3"):
                idx = futures[future]
                t3_results[idx] = future.result()

    # 6. Final Summary
    final_correct = sum(1 for i in range(len(questions)) if t1_results[i] or t3_results[i])
    final_acc = final_correct / len(questions)
    fixes = sum(t3_results)
    
    print("\n" + "="*60)
    print(f"📊 ITERATIVE FIXED FEEDBACK SUMMARY")
    print(f"Model: {args.model_path}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print("-" * 60)
    print(f"Total Samples:       {len(questions)}")
    print(f"Turn 1 Accuracy:     {t1_acc:>10.2%}")
    print(f"Self-Corrections:    {fixes:>10} (fixed {fixes/len(incorrect_indices) if incorrect_indices else 0:.1%} of errors)")
    print(f"Final Accuracy:      {final_acc:>10.2%}")
    print("="*60)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results = []
    for i in range(len(questions)):
        results.append({
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "y0": y0_list[i],
            "y1": y1_list[i],
            "t1_correct": t1_results[i],
            "t3_correct": t3_results[i],
            "final_correct": t1_results[i] or t3_results[i]
        })

    with open(args.output_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    run_eval()