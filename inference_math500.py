import argparse
import json
import os
import asyncio
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
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

def run_inference():
    parser = argparse.ArgumentParser(description="Multi-turn inference for Math500")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Path or HF ID of the model")
    parser.add_argument("--data_path", type=str, default="/home/ssmurali/mlmt/data/mlmt/math/test.parquet", help="Path to Math500 parquet")
    parser.add_argument("--turns", type=int, default=3, help="Total number of turns (T = 2K+1)")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--use_semantic", action="store_true", default=True, help="Use LLM-based semantic scoring")
    parser.add_argument("--no_semantic", action="store_false", dest="use_semantic", help="Disable LLM-based semantic scoring")
    args = parser.parse_args()

    if args.turns % 2 == 0:
        raise ValueError("Number of turns T must be odd (1, 3, 5, ...)")

    if args.output_path is None:
        args.output_path = f"math500_T{args.turns}_results.jsonl"

    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens)

    print(f"Loading data: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    if args.num_samples:
        df = df.head(args.num_samples)

    results = []
    
    # Pre-extract questions and ground truths
    questions = [extract_question(row['prompt']) for _, row in df.iterrows()]
    ground_truths = [row['reward_model']['ground_truth'] for _, row in df.iterrows()]
    
    # Store state for each sample: current_solution, current_feedback
    current_solutions = [None] * len(questions)
    current_feedbacks = [None] * len(questions)
    
    # Store responses for ALL turns
    all_responses = {} 

    # Multi-turn Loop
    for t in range(1, args.turns + 1):
        print(f"\n--- Starting Turn {t} ---")
        prompts = []
        
        if t == 1:
            # Turn 1: Initial Solve
            for q in questions:
                prompts.append(prepare_mlmt_turn1_prompt(q))
        elif t % 2 == 0:
            # Feedback Turn (Even turns: 2, 4, 6...)
            for q, soln in zip(questions, current_solutions):
                prompts.append(prepare_mlmt_feedback_prompt(q, soln))
        else:
            # Refinement Turn (Odd turns > 1: 3, 5, 7...)
            for q, feedback in zip(questions, current_feedbacks):
                prompts.append(prepare_mlmt_refinement_prompt(q, feedback))

        # Generate responses in batch
        outputs = llm.generate(prompts, sampling_params)
        responses = [out.outputs[0].text for out in outputs]
        all_responses[t] = responses
        
        # Update state
        if t % 2 == 1:
            current_solutions = responses
        else:
            current_feedbacks = responses

    # Compute accuracy for ALL solve turns
    print("\n" + "="*40)
    print(f"{'TURN':<10} | {'TYPE':<10} | {'ACCURACY':<10}")
    print("-" * 40)
    
    # Pre-compute all scores for summary and logging using async parallel processing
    turn_scores = {} # {turn: [scores]}
    solve_turns = sorted([t for t in all_responses.keys() if t % 2 == 1])
    
    for t in solve_turns:
        print(f"Scoring Turn {t}...")
        solutions = all_responses[t]
        scores = asyncio.run(compute_all_scores(solutions, ground_truths, args.use_semantic))
        turn_scores[t] = [float(s) for s in scores]
        
        acc = sum(1 for s in scores if s > 0) / len(questions) if questions else 0
        print(f"{t:<10} | {'Solve':<10} | {acc:>10.2%}")
    print("="*40)

    # Prepare final results for saving
    results = []
    for i in range(len(questions)):
        sample_res = {
            "index": i,
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "turns": {}
        }
        for t in sorted(all_responses.keys()):
            res_text = all_responses[t][i]
            turn_data = {"content": res_text}
            if t in turn_scores:
                turn_data["score"] = turn_scores[t][i]
                turn_data["type"] = "solve"
            else:
                turn_data["type"] = "feedback"
            
            sample_res["turns"][t] = turn_data
        
        results.append(sample_res)

    # Save to JSONL
    with open(args.output_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    print(f"\nResults saved to {args.output_path}")

if __name__ == "__main__":
    run_inference()

