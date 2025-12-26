#!/usr/bin/env python3
"""
Reward Re-evaluation Script

Goes through all result files in a directory and re-evaluates whether the terminal rewards are correct.
Uses CMU Gateway GPT-4o-mini for soft semantic evaluation of answer vs ground truth.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import openai
from openai import OpenAI

# CMU Gateway configuration
CMU_GATEWAY_BASE_URL = "https://ai-gateway.andrew.cmu.edu"
GATEWAY_API_KEY = "sk-sejf_6tjIhvazsxWCH-mZg"
MODEL_NAME = "gpt-4o-mini-2024-07-18"

def get_cmu_client():
    """Initialize OpenAI client for CMU Gateway."""
    return OpenAI(
        api_key=GATEWAY_API_KEY,
        base_url=CMU_GATEWAY_BASE_URL
    )

# Soft evaluation prompt - more lenient than the original
SOFT_EVALUATION_PROMPT = """
Please evaluate if the predicted answer is semantically equivalent to the ground truth answer.

**EVALUATION CRITERIA:**
- Answers are CORRECT if they contain the same core factual information
- The predicted answer should convey the same meaning as the ground truth
- Minor differences in wording, capitalization, punctuation are OK
- Extra context or elaboration is OK as long as the core fact is correct
- Partial matches that capture the essential information are acceptable
- Only mark INCORRECT if the facts are genuinely different or contradictory

**EXAMPLES:**
- Ground truth: "more than three weeks"
  Predicted: "More than three weeks" → CORRECT (same meaning)
- Ground truth: "Searchlight 2023 Childlight Annual Flagship Report"
  Predicted: "The title is 'Searchlight 2023 – Childlight Annual Flagship Report'" → CORRECT (contains the core fact)
- Ground truth: "Philippines Electric Vehicle Market (2025-2031) Industry Report"
  Predicted: "6Wresearch provides reports including 'Philippines Electric Vehicle Market (2025-2031) Industry Report'" → CORRECT (mentions the exact title)

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {predicted_answer}

Please respond with a JSON object:
{{
"rationale": "brief explanation of your evaluation",
"judgement": "correct" or "incorrect",
"confidence": "high" or "medium" or "low"
}}
"""

def evaluate_answer_soft(question: str, predicted_answer: str, ground_truth: str) -> Tuple[bool, str]:
    """
    Soft evaluation of answer vs ground truth using CMU Gateway GPT-4o-mini.

    Returns:
        (is_correct: bool, rationale: str)
    """
    client = get_cmu_client()

    prompt = SOFT_EVALUATION_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        predicted_answer=predicted_answer
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=500,
                top_p=0.9
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON response
            try:
                result = json.loads(content)
                judgement = result.get('judgement', 'incorrect').lower()
                rationale = result.get('rationale', 'No rationale provided')

                is_correct = judgement == 'correct'
                return is_correct, rationale

            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract judgement from text
                content_lower = content.lower()
                if 'correct' in content_lower and 'incorrect' not in content_lower:
                    return True, f"Text-based evaluation: {content[:100]}"
                else:
                    return False, f"Text-based evaluation: {content[:100]}"

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API call failed (attempt {attempt + 1}), retrying: {e}")
                time.sleep(1 * (2 ** attempt))  # Exponential backoff
            else:
                print(f"API call failed after {max_retries} attempts: {e}")
                return False, f"API Error: {str(e)}"

    return False, "Failed to evaluate after all retries"

def process_result_file(filepath: Path, question: str = None) -> Tuple[bool, str, int]:
    """
    Process a single result file.

    Returns:
        (should_be_correct: bool, rationale: str, current_reward: int)
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        predicted_answer = data.get('answer', '').strip()
        ground_truth = data.get('ground_truth', '').strip()
        current_reward = data.get('reward_sum', 0)

        # Skip if no answer or ground truth
        if not predicted_answer or not ground_truth:
            return False, "Missing answer or ground truth", current_reward

        # Evaluate using soft criteria
        should_be_correct, rationale = evaluate_answer_soft(
            question or "Unknown question",
            predicted_answer,
            ground_truth
        )

        return should_be_correct, rationale, current_reward

    except Exception as e:
        return False, f"Error processing file: {str(e)}", 0

def update_reward_in_file(filepath: Path, new_reward: int) -> bool:
    """Update the reward_sum in a result file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        data['reward_sum'] = new_reward

        # Update individual rewards to match the sum (simplified)
        if 'rewards' in data and data['rewards']:
            # Set the last reward to achieve the desired sum
            total_other_rewards = sum(data['rewards'][:-1]) if len(data['rewards']) > 1 else 0
            data['rewards'][-1] = new_reward - total_other_rewards

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return True

    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Re-evaluate rewards in DeepResearch result files')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing result JSON files')
    parser.add_argument('--fix_incorrect', action='store_true',
                       help='Actually fix incorrect rewards (otherwise just report)')
    parser.add_argument('--question', type=str, default=None,
                       help='Question text (optional, used for context)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Directory {input_dir} does not exist")
        return

    # Find all result files
    result_files = list(input_dir.glob("result_*.json"))
    print(f"Found {len(result_files)} result files")

    correct_count = 0
    incorrect_count = 0
    fixed_count = 0

    for filepath in sorted(result_files):
        print(f"\nProcessing {filepath.name}...")

        should_be_correct, rationale, current_reward = process_result_file(filepath, args.question)

        is_currently_correct = current_reward > 0

        if should_be_correct and not is_currently_correct:
            print(f"❌ INCORRECT: Should be correct but reward is {current_reward}")
            print(f"   Rationale: {rationale}")
            incorrect_count += 1

            if args.fix_incorrect:
                if update_reward_in_file(filepath, 1):
                    print("   ✅ Fixed: Set reward_sum to 1")
                    fixed_count += 1
                else:
                    print("   ❌ Failed to fix")

        elif not should_be_correct and is_currently_correct:
            print(f"⚠️  OVER-REWARDED: Should be incorrect but reward is {current_reward}")
            print(f"   Rationale: {rationale}")
            incorrect_count += 1

            if args.fix_incorrect:
                if update_reward_in_file(filepath, 0):
                    print("   ✅ Fixed: Set reward_sum to 0")
                    fixed_count += 1
                else:
                    print("   ❌ Failed to fix")

        elif should_be_correct and is_currently_correct:
            print(f"✅ CORRECT: Already correctly rewarded ({current_reward})")
            correct_count += 1

        else:  # not should_be_correct and not is_currently_correct
            print(f"✅ CORRECT: Already correctly not rewarded ({current_reward})")
            correct_count += 1

    print("\n=== SUMMARY ===")
    print(f"Total files processed: {len(result_files)}")
    print(f"Correct evaluations: {correct_count}")
    print(f"Incorrect evaluations: {incorrect_count}")
    if args.fix_incorrect:
        print(f"Files fixed: {fixed_count}")

if __name__ == "__main__":
    main()
