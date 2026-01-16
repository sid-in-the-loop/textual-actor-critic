import argparse
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing
import sys
import io
import re
import ast

# Set this to avoid fork warnings from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to sys.path to import mlmt_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_system.multi_turn_rollout.mlmt_utils import (
    prepare_mlmt_code_turn1_prompt,
    prepare_mlmt_code_feedback_prompt,
    prepare_mlmt_code_refinement_prompt
)

# --- Code execution worker (same as mlmt_envs.py) ---
def _execute_code_worker(code, test, entry_point, queue):
    """Worker function for executing code in a separate process."""
    import sys
    import io
    # Redirect stdout/stderr to avoid cluttering Ray logs
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    scope = {}
    try:
        # Pre-compile the code to separate syntax errors from execution errors
        compiled_code = compile(code, '<string>', 'exec')
        # Use the same dictionary for both globals and locals to ensure 
        # that functions defined in the code can access the imports.
        exec(compiled_code, scope)
        
        # Execute the test string which defines 'check'
        exec(test, scope)
        
        # Call check(candidate)
        if 'check' in scope:
            if entry_point in scope:
                scope['check'](scope[entry_point])
                queue.put((True, ""))
            else:
                queue.put((False, f"entry_point_not_found: {entry_point}"))
        else:
            queue.put((False, "check_func_not_found"))
    except Exception as exc:
        queue.put((False, f"execution_error: {exc}"))

def run_code_tests(code_str, test, entry_point, timeout=10.0):
    """
    Execute candidate code against HumanEval test.
    Returns (passed: bool, error: str)
    """
    if not isinstance(code_str, str):
        return False, "invalid_code_type"

    # Extract code from block if present
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    match = re.search(pattern, code_str, flags=re.IGNORECASE)
    code = match.group(1).strip() if match else code_str.strip()
    
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_code_worker,
        args=(code, test, entry_point, queue)
    )
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return False, "timeout"
    
    if queue.empty():
        return False, "process_crashed"
    
    passed, error = queue.get()
    return passed, error

def run_humaneval_eval():
    parser = argparse.ArgumentParser(description="HumanEval Iterative Evaluation")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--low_lora", type=str, required=True, help="Path to actor/lora_adapter")
    parser.add_argument("--high_lora", type=str, required=True, help="Path to high_actor/lora_adapter")
    parser.add_argument("--output_path", type=str, default="results/humaneval_iterative.jsonl")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"🚀 Loading vLLM with Base: {args.base_model}")
    print(f"   Low LoRA: {args.low_lora}")
    print(f"   High LoRA: {args.high_lora}")
    
    # Load base model and enable LoRA
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_loras=2,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=4096
    )
    
    low_lora_req = LoRARequest("low_actor", 1, args.low_lora)
    high_lora_req = LoRARequest("high_actor", 2, args.high_lora)
    
    # Parity Config
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=1024,
        n=1,
        stop=["<|eot_id|>", "<|end_of_text|>", "```\n"]
    )

    print("Loading HumanEval data...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # We do inference in turns for all samples to utilize batching
    samples = []
    for item in dataset:
        # INJECT HINT: Ensure function name matches HumanEval entry_point
        prompt_with_hint = f"{item['prompt']}\nYour function should be named `{item['entry_point']}`."
        samples.append({
            "task_id": item['task_id'],
            "question": prompt_with_hint,
            "test": item['test'],
            "entry_point": item['entry_point'],
            "turns": {},
            "resolved": False
        })

    # --- Turn 1: Solve (Low Actor) ---
    print(f"\n--- Turn 1: Solving (Total: {len(samples)}) ---")
    t1_prompts = []
    for s in samples:
        raw_p = prepare_mlmt_code_turn1_prompt(s['question'])
        messages = [{"role": "user", "content": raw_p}]
        chat_p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t1_prompts.append(chat_p)

    outputs = llm.generate(t1_prompts, sampling_params, lora_request=low_lora_req)
    for s, out in zip(samples, outputs):
        s['turns']['1'] = {"content": out.outputs[0].text, "type": "actor_solve"}

    print("🔍 Evaluating Turn 1...")
    for s in tqdm(samples, desc="Eval T1"):
        passed, error = run_code_tests(s['turns']['1']['content'], s['test'], s['entry_point'])
        s['turns']['1']['passed'] = passed
        s['turns']['1']['error'] = error
        if passed:
            s['resolved'] = True

    # --- Turn 2: Feedback (High Actor) ---
    # Only refine samples that failed Turn 1
    active_samples = [s for s in samples if not s['resolved']]
    
    if active_samples:
        print(f"\n--- Turn 2: Feedback (Active: {len(active_samples)}) ---")
        t2_prompts = []
        for s in active_samples:
            t1_sol = s['turns']['1']['content']
            t1_err = s['turns']['1']['error']
            raw_p = prepare_mlmt_code_feedback_prompt(s['question'], t1_sol, error=t1_err)
            messages = [{"role": "user", "content": raw_p}]
            chat_p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            t2_prompts.append(chat_p)

        outputs = llm.generate(t2_prompts, sampling_params, lora_request=high_lora_req)
        for s, out in zip(active_samples, outputs):
            s['turns']['2'] = {"content": out.outputs[0].text, "type": "mentor_feedback"}

        # --- Turn 3: Refine (Low Actor) ---
        print(f"\n--- Turn 3: Refining (Active: {len(active_samples)}) ---")
        t3_prompts = []
        for s in active_samples:
            t1_sol = s['turns']['1']['content']
            t2_feedback = s['turns']['2']['content']
            raw_p = prepare_mlmt_code_refinement_prompt(s['question'], t1_sol, t2_feedback)
            messages = [{"role": "user", "content": raw_p}]
            chat_p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            t3_prompts.append(chat_p)

        outputs = llm.generate(t3_prompts, sampling_params, lora_request=low_lora_req)
        for s, out in zip(active_samples, outputs):
            s['turns']['3'] = {"content": out.outputs[0].text, "type": "actor_refine"}

        print("🔍 Evaluating Turn 3...")
        for s in tqdm(active_samples, desc="Eval T3"):
            passed, error = run_code_tests(s['turns']['3']['content'], s['test'], s['entry_point'])
            s['turns']['3']['passed'] = passed
            s['turns']['3']['error'] = error
            if passed:
                s['resolved'] = True
    else:
        print("\n✅ All samples resolved in Turn 1. Skipping Turns 2 & 3.")

    # Summary
    total = len(samples)
    t1_correct = sum(1 for s in samples if s['turns']['1']['passed'])
    total_correct = sum(1 for s in samples if s['resolved'])
    self_corrections = total_correct - t1_correct
    
    print("\n" + "="*60)
    print(f"📊 ITERATIVE HUMANEVAL EVALUATION (RESOLVED LOGIC)")
    print("-" * 60)
    print(f"Total Samples:       {total}")
    print(f"Turn 1 Pass@1:       {t1_correct/total:>10.2%}")
    print(f"Cumulative Pass@1:   {total_correct/total:>10.2%}")
    print(f"Self-Corrections:    {self_corrections} cases (+{self_corrections/total:.2%})")
    print("="*60)

    # Save results
    with open(args.output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"\n✅ Results saved to {args.output_path}")

if __name__ == "__main__":
    run_humaneval_eval()
