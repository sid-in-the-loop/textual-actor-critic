import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from openai import OpenAI
import pickle
import re

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
JSONL_PATH = "results/mlmt_math500_T3_iterative.jsonl"
OUTPUT_DIR = "experiments/outputs"
OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_PER_EXP = 10000
LEARNING_RATE = 1e-5

def extract_answer(text):
    """Extracts answer from \boxed{} or the last number."""
    boxed = re.findall(r'\\boxed\{(.*?)\}', text)
    if boxed:
        return boxed[-1]
    words = text.split()
    if words:
        return words[-1].strip('.')
    return ""

def get_samples():
    correct_samples = []
    wrong_samples = []
    
    with open(JSONL_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            turn1 = data['turns'].get('1')
            if not turn1:
                continue
                
            sample = {
                'prompt': data['question'],
                'trajectory': turn1['content'],
                'ground_truth': data['ground_truth'],
                'index': data['index'],
                'answer': extract_answer(turn1['content'])
            }
            
            if data['resolved_at_turn'] == 1:
                if len(correct_samples) < 100:
                    correct_samples.append(sample)
            elif data['resolved_at_turn'] is None:
                if len(wrong_samples) < 100:
                    wrong_samples.append(sample)
            
            if len(correct_samples) >= 100 and len(wrong_samples) >= 100:
                break
                
    return correct_samples, wrong_samples

def find_turning_token_gpt(prompt, trajectory, ground_truth):
    system_prompt = "You are an expert math tutor. Identify the EXACT piece of text where the reasoning first goes wrong in the provided trajectory."
    user_prompt = f"Prompt: {prompt}\nGround Truth: {ground_truth}\nTrajectory: {trajectory}\n\nReturn ONLY the exact few words where the error starts. If the trajectory is correct, return 'CORRECT'."
    
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT error: {e}")
        return "ERROR"

def get_token_index(tokenizer, trajectory, error_text):
    if not error_text or error_text == "CORRECT" or error_text == "ERROR":
        return -1
    tokens = tokenizer.encode(trajectory, add_special_tokens=False)
    error_tokens = tokenizer.encode(error_text, add_special_tokens=False)
    for i in range(len(tokens) - len(error_tokens) + 1):
        if tokens[i:i+len(error_tokens)] == error_tokens:
            return i
    return -1

def compute_loss(method, log_probs, reward, turning_idx):
    if reward == 1:
        return -1.0 * log_probs.sum()
    if method == 'normal':
        return torch.tensor(0.0, device=log_probs.device, requires_grad=True)
    if turning_idx == -1:
        turning_idx = len(log_probs) - 1
    if method == 'masking':
        return -1.0 * log_probs[:turning_idx].sum()
    if method == 'negative':
        loss_good = -1.0 * log_probs[:turning_idx].sum()
        loss_bad = 1.0 * log_probs[turning_idx:].sum()
        return loss_good + loss_bad
    return torch.tensor(0.0, device=log_probs.device, requires_grad=True)

def find_subsequence(full, sub):
    n, m = len(full), len(sub)
    for i in range(n - m + 1):
        if full[i:i+m] == sub:
            return i
    return -1

def run_experiment(method, n_samples, sample_pool, cat_name):
    exp_id = f"{method}_{cat_name}_n{n_samples}"
    print(f"\n{'='*60}\nRUNNING: {exp_id}\n{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    selected_samples = sample_pool[:n_samples]
    processed = []
    
    print(f"Preprocessing {n_samples} samples...")
    for s in selected_samples:
        reward = 1.0 if cat_name == "correct" else 0.0
        turning_idx = -1
        if reward == 0.0:
            error_text = find_turning_token_gpt(s['prompt'], s['trajectory'], s['ground_truth'])
            turning_idx = get_token_index(tokenizer, s['trajectory'], error_text)
            print(f"Index {s['index']} | Turning Token: {error_text} at {turning_idx}")
        
        prompt_ids = tokenizer.encode(str(s['prompt']), add_special_tokens=True)
        traj_ids = tokenizer.encode(s['trajectory'], add_special_tokens=False)
        
        ans_text = s['answer']
        ans_gt = extract_answer(s['ground_truth'])
        ans_ids = tokenizer.encode(ans_text, add_special_tokens=False)
        gt_ids = tokenizer.encode(ans_gt, add_special_tokens=False)
        
        ans_start_idx = find_subsequence(traj_ids, ans_ids)
        
        processed.append({
            'input_ids': torch.tensor(prompt_ids + traj_ids).to(DEVICE),
            'labels': torch.tensor([-100]*len(prompt_ids) + traj_ids).to(DEVICE),
            'reward': reward,
            'turning_idx': turning_idx,
            'prompt_len': len(prompt_ids),
            'traj_len': len(traj_ids),
            'gt_token_id': gt_ids[0] if gt_ids else None,
            'ans_token_id': ans_ids[0] if ans_ids else None,
            'ans_token_idx': ans_start_idx 
        })

    metrics = defaultdict(list)
    initial_probs = None
    
    for step in range(STEPS_PER_EXP):
        optimizer.zero_grad()
        
        target_s = processed[0] 
        
        batch_size = n_samples
        indices = np.random.choice(len(processed), batch_size, replace=True)
        
        total_loss = 0
        for idx in indices:
            s = processed[idx]
            outputs = model(s['input_ids'].unsqueeze(0), labels=s['labels'].unsqueeze(0))
            logits = outputs.logits[0, s['prompt_len']-1:-1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            target_ids = s['input_ids'][s['prompt_len']:]
            token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            
            loss = compute_loss(method, token_log_probs, s['reward'], s['turning_idx'])
            if loss.requires_grad:
                total_loss += loss / batch_size

        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            with torch.no_grad():
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                metrics['grad_norm'].append(grad_norm)
            optimizer.step()

        with torch.no_grad():
            pool_p_correct = []
            pool_p_wrong = []
            
            track_pool = processed[:min(10, n_samples)]
            for s in track_pool:
                outputs = model(s['input_ids'].unsqueeze(0))
                logits = outputs.logits[0, s['prompt_len']-1:-1, :]
                probs = torch.softmax(logits, dim=-1)
                
                if s['ans_token_idx'] != -1 and s['ans_token_id'] is not None and s['gt_token_id'] is not None:
                    p_w = probs[s['ans_token_idx'], s['ans_token_id']].item()
                    p_c = probs[s['ans_token_idx'], s['gt_token_id']].item()
                    pool_p_correct.append(p_c)
                    pool_p_wrong.append(p_w)
            
            if pool_p_correct:
                metrics['p_correct_answer'].append(np.mean(pool_p_correct))
                metrics['p_wrong_answer'].append(np.mean(pool_p_wrong))
                metrics['prob_ratio'].append(np.mean(pool_p_correct) / (np.mean(pool_p_wrong) + 1e-8))

            outputs = model(target_s['input_ids'].unsqueeze(0))
            logits = outputs.logits[0, target_s['prompt_len']-1:-1, :]
            probs = torch.softmax(logits, dim=-1)
            target_ids = target_s['input_ids'][target_s['prompt_len']:]
            token_probs = probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).cpu().numpy()
            
            if initial_probs is None:
                initial_probs = token_probs
            
            metrics['token_probs'].append(token_probs.tolist())
            metrics['delta_prob'].append((token_probs - initial_probs).tolist())
            metrics['total_loss'].append(total_loss.item() if isinstance(total_loss, torch.Tensor) else 0)

        if step % 1000 == 0:
            p_c = metrics['p_correct_answer'][-1] if metrics['p_correct_answer'] else 0
            p_w = metrics['p_wrong_answer'][-1] if metrics['p_wrong_answer'] else 0
            print(f"Step {step} | Loss: {metrics['total_loss'][-1]:.4f} | Avg P(Correct): {p_c:.4f} | Avg P(Wrong): {p_w:.4f}")

    with open(f"{OUTPUT_DIR}/{exp_id}_metrics.pkl", 'wb') as f:
        pickle.dump(dict(metrics), f)
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, wrong_pool = get_samples()
    
    for method in ['normal', 'masking', 'negative']:
        for n in [1, 10, 100]:
            run_experiment(method, n, wrong_pool, "wrong")
