import argparse
import json
import os
import asyncio
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from agent_system.multi_turn_rollout.mlmt_utils import (
    prepare_mlmt_turn1_prompt,
    prepare_mlmt_feedback_prompt,
    prepare_mlmt_refinement_prompt
)
from verl.utils.reward_score.math import compute_score, compute_score_async

def extract_question(prompt_col):
    if isinstance(prompt_col, (list, pd.Series)) and len(prompt_col) > 0:
        if isinstance(prompt_col[0], dict) and 'content' in prompt_col[0]:
            return prompt_col[0]['content']
    return str(prompt_col)

async def compute_all_scores(solutions, ground_truths, use_semantic):
    """Compute scores for a batch of solutions in parallel."""
    tasks = [compute_score_async(sol, gt, use_semantic=use_semantic) for sol, gt in zip(solutions, ground_truths)]
    return await asyncio.gather(*tasks)

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

def run_inference():
    parser = argparse.ArgumentParser(description="Multi-turn Actor-Guider inference for Math500")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        help="Base model (always used for High Level Guider)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        help="Actor model (Low Level - can be adapter or base)")
    parser.add_argument("--data_path", type=str, default="/home/ssmurali/mlmt/data/mlmt/math/test.parquet", help="Path to Math500 parquet")
    parser.add_argument("--turns", type=int, default=3, help="Total number of turns (T = 2K+1)")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--use_semantic", action="store_true", default=True, help="Use LLM-based semantic scoring")
    parser.add_argument("--no_semantic", action="store_false", dest="use_semantic", help="Disable LLM-based semantic scoring")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token for private models")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key for evals")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    if args.use_semantic and "OPENAI_API_KEY" not in os.environ:
        print("⚠️ Warning: OPENAI_API_KEY not found. Semantic scoring may fail.")

    if args.turns % 2 == 0:
        raise ValueError("Number of turns T must be odd (1, 3, 5, ...)")

    # Set up results directory and output path
    os.makedirs("results", exist_ok=True)
    if args.output_path is None:
        if args.turns == 1:
            args.output_path = "results/HLF_LLF.csv"
        else:
            args.output_path = f"results/math500_T{args.turns}_actor_guider_results.jsonl"

    print(f"Loading data: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    if args.num_samples:
        df = df.head(args.num_samples)

    questions = [extract_question(row['prompt']) for _, row in df.iterrows()]
    ground_truths = [row['reward_model']['ground_truth'] for _, row in df.iterrows()]

    # Engine initialization (always use vLLM for speed)
    use_lora = (args.model != args.base_model)
    print(f"🚀 Loading vLLM Engine (TP={args.tp})")
    print(f"🚀 Guider Model (Base): {args.base_model}")
    print(f"🚀 Actor Model (Target): {args.model}")

    llm = LLM(
        model=args.base_model, 
        enable_lora=use_lora, 
        max_loras=1 if use_lora else 0,
        trust_remote_code=True, 
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tp,
        dtype="bfloat16" # Ensure bf16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    actor_lora = LoRARequest("actor_lora", 1, args.model) if use_lora else None
    
    # Sampling params
    # For Turn 1 benchmark, use deterministic settings
    if args.turns == 1:
        params_turn1 = SamplingParams(temperature=0.0, max_tokens=2048, n=1)

    else:
        params_turn1 = SamplingParams(temperature=args.temperature, max_tokens=2048, n=1)
        
    params_turn2 = SamplingParams(temperature=args.temperature, max_tokens=512, n=1)
    params_turn3 = SamplingParams(temperature=args.temperature, max_tokens=2048, n=1)

    current_solutions = [None] * len(questions)
    current_feedbacks = [None] * len(questions)
    all_responses = {} 

    for t in range(1, args.turns + 1):
        is_actor_turn = (t % 2 == 1)
        role = "Actor (Low-Level)" if is_actor_turn else "Guider (High-Level)"
        print(f"\n--- Turn {t} ({role}) ---")
        
        prompts = []
        if t == 1:
            if args.turns == 1:
                # Use the benchmark-faithful prompt template
                prompts = [build_benchmark_prompt(tokenizer, q) for q in questions]
            else:
                # Use the MLMT turn 1 template
                prompts = [prepare_mlmt_turn1_prompt(q) for q in questions]
            current_params = params_turn1
        elif not is_actor_turn:
            prompts = [prepare_mlmt_feedback_prompt(q, soln) for q, soln in zip(questions, current_solutions)]
            current_params = params_turn2
        else:
            prompts = [prepare_mlmt_refinement_prompt(q, sol, feedback) for q, sol, feedback in zip(questions, current_solutions, current_feedbacks)]
            current_params = params_turn3

        current_lora = actor_lora if is_actor_turn else None
        outputs = llm.generate(prompts, current_params, lora_request=current_lora)
        responses = [out.outputs[0].text for out in outputs]
        all_responses[t] = responses
        
        if is_actor_turn:
            current_solutions = responses
        else:
            current_feedbacks = responses

    # Saving Results
    if args.turns == 1:
        df["model_output"] = all_responses[1]
        df.to_csv(args.output_path, index=False)
        print(f"✅ Benchmark results saved to {args.output_path}")
    else:
        # Scoring and saving for multi-turn...
        turn_scores = {}
        solve_turns = sorted([t for t in all_responses.keys() if t % 2 == 1])
        for t in solve_turns:
            solutions = all_responses[t]
            scores = asyncio.run(compute_all_scores(solutions, ground_truths, args.use_semantic))
            turn_scores[t] = [float(s) for s in scores]

        results = []
        for i in range(len(questions)):
            sample_res = {"index": i, "question": questions[i], "ground_truth": ground_truths[i], "turns": {}}
            for t in sorted(all_responses.keys()):
                turn_data = {"content": all_responses[t][i]}
                if t % 2 == 1:
                    turn_data.update({"type": "actor_solve", "score": turn_scores[t][i]})
                else:
                    turn_data.update({"type": "guider_feedback"})
                sample_res["turns"][t] = turn_data
            results.append(sample_res)

        # Save JSONL
        jsonl_path = args.output_path if args.output_path.endswith('.jsonl') else args.output_path.replace('.csv', '.jsonl')
        with open(jsonl_path, "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
        
        # Save flattened CSV for easy viewing
        csv_path = args.output_path if args.output_path.endswith('.csv') else args.output_path.replace('.jsonl', '.csv')
        flattened_data = []
        for res in results:
            flat_row = {
                "question": res["question"],
                "ground_truth": res["ground_truth"]
            }
            for t, turn_data in res["turns"].items():
                flat_row[f"turn{t}_output"] = turn_data["content"]
                if "score" in turn_data:
                    flat_row[f"turn{t}_score"] = turn_data["score"]
            flattened_data.append(flat_row)
        
        pd.DataFrame(flattened_data).to_csv(csv_path, index=False)
        print(f"\n✅ Multi-turn results saved to:\n   - JSONL: {jsonl_path}\n   - CSV:   {csv_path}")

if __name__ == "__main__":
    run_inference()
