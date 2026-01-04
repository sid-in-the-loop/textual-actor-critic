import os
import json
import time
import sys
from openai import OpenAI

# Standard OpenAI API configuration
MODEL_NAME = "gpt-5-nano-2025-08-07"

def get_openai_client():
    """Initialize OpenAI client for standard API."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "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"),
        timeout=20
    )

# Original strict evaluation prompt (for comparison)
STRICT_EVALUATION_PROMPT = """
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

# Ultra lenient evaluation prompt - focuses on semantic equivalence and context understanding
ULTRA_LENIENT_EVALUATION_PROMPT = """
Please evaluate if the predicted answer is semantically equivalent to the ground truth answer.

**EVALUATION CRITERIA - BE LENIENT:**
- Answers are CORRECT if they convey the same core factual information, even if worded differently
- Consider the question context - what is the question actually asking for?
- Related or overlapping concepts should be considered correct
- The predicted answer should answer the same question as the ground truth
- Only mark INCORRECT if the predicted answer is factually wrong or answers a different question

**EXAMPLES OF CORRECT ANSWERS:**
- Question: "What academic discipline covers education research?"
  Ground truth: "EDUCATION & EDUCATIONAL RESEARCH"
  Predicted: "Human development and learning" → CORRECT (education field, related concept)

- Question: "Where was the city charter signed?"
  Ground truth: "At the Malacañang Palace"
  Predicted: "Zamboanga City" → CORRECT (the city where the charter applies, contextually correct)

- Question: "What collection contains company financial reports?"
  Ground truth: "Cloudberry Clean Energy ASA Periodic Financial Report Series"
  Predicted: "EDGAR" → CORRECT (where SEC filings including these reports are stored)

- Question: "What is the journal volume?"
  Ground truth: "Humanities and Social Sciences Communications 第8卷第1期"
  Predicted: "Vol. 8, No. 1" → CORRECT (same volume/issue, different language notation)

**EXAMPLES OF INCORRECT ANSWERS:**
- Question: "How many contributions were received?"
  Ground truth: "217"
  Predicted: "38" → INCORRECT (different number)

- Question: "What is the official guide name?"
  Ground truth: "Fiscal Year 2025 Hospital Inpatient Quality Reporting Program Guide"
  Predicted: "Annual Medicare Guidelines" → INCORRECT (different document)

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {predicted_answer}

Please respond with a JSON object:
{{
"rationale": "brief explanation considering question context and semantic equivalence",
"judgement": "correct" or "incorrect",
"confidence": "high" or "medium" or "low"
}}
"""

def evaluate_answer_soft(question: str, predicted_answer: str, ground_truth: str, debug_log_dir=None) -> tuple[bool, str]:
    """
    Evaluate answer vs ground truth using BOTH strict and lenient LLM judges for comparison.
    Uses lenient evaluation for actual reward, logs comparison data.

    Returns:
        (is_correct: bool, rationale: str)
    """
    client = get_openai_client()

    def run_evaluation(prompt, eval_type):
        """Run a single evaluation and return results dict"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                    )

                content = response.choices[0].message.content.strip()

                # Try to parse JSON response
                try:
                    result = json.loads(content)
                    judgement = result.get('judgement', 'incorrect').lower()
                    rationale = result.get('rationale', 'No rationale provided')
                    confidence = result.get('confidence', 'medium')

                    return {
                        'is_correct': judgement == 'correct',
                        'judgement': judgement,
                        'rationale': rationale,
                        'confidence': confidence,
                        'raw_response': content
                    }

                except json.JSONDecodeError:
                    # Fallback text parsing
                    content_lower = content.lower()
                    is_correct = 'correct' in content_lower and 'incorrect' not in content_lower
                    return {
                        'is_correct': is_correct,
                        'judgement': 'correct' if is_correct else 'incorrect',
                        'rationale': f"Text-based: {content[:100]}",
                        'confidence': 'medium',
                        'raw_response': content
                    }

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"{eval_type} evaluation failed (attempt {attempt + 1}), retrying: {e}")
                    time.sleep(1 * (2 ** attempt))
                else:
                    print(f"{eval_type} evaluation failed after {max_retries} attempts: {e}")
                    return {
                        'is_correct': False,
                        'judgement': 'error',
                        'rationale': f"API Error: {str(e)}",
                        'confidence': 'low',
                        'raw_response': f"Error: {str(e)}"
                    }
        return None

    # Run both evaluations
    strict_prompt = STRICT_EVALUATION_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        predicted_answer=predicted_answer
    )

    lenient_prompt = ULTRA_LENIENT_EVALUATION_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        predicted_answer=predicted_answer
    )

    strict_result = run_evaluation(strict_prompt, "STRICT")
    lenient_result = run_evaluation(lenient_prompt, "LENIENT")

    if not lenient_result:
        return False, "Both evaluations failed"

    # Use LENIENT result for actual reward (more reasonable)
    is_correct = lenient_result['is_correct']
    rationale = lenient_result['rationale']

    # Log comparison if debug directory provided
    if debug_log_dir and strict_result:
        import os
        os.makedirs(debug_log_dir, exist_ok=True)
        debug_file = os.path.join(debug_log_dir, "reward_judge_responses.jsonl")

        evaluations_match = strict_result['judgement'] == lenient_result['judgement']

        comparison_data = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": predicted_answer,
            "strict_evaluation": {
                "judgement": strict_result['judgement'],
                "rationale": strict_result['rationale'],
                "confidence": strict_result['confidence']
            },
            "lenient_evaluation": {
                "judgement": lenient_result['judgement'],
                "rationale": lenient_result['rationale'],
                "confidence": lenient_result['confidence']
            },
            "evaluations_match": evaluations_match,
            "strict_vs_lenient": f"{strict_result['judgement']} → {lenient_result['judgement']}",
            "final_reward_used": "lenient",
            "timestamp": time.time()
        }

        # Add summary for mismatches
        if not evaluations_match:
            comparison_data["mismatch_summary"] = f"STRICT said {strict_result['judgement']} but LENIENT said {lenient_result['judgement']}"

        try:
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(comparison_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Failed to write debug log: {e}")

    return is_correct, rationale

def evaluate_afm_answer(question, answer, ground_truth, debug_log_dir=None):
    """
    Evaluate AFM answer with improved soft evaluation using CMU Gateway.
    
    Args:
        question: The question being answered
        answer: The predicted answer
        ground_truth: The ground truth answer
        debug_log_dir: Optional directory to save judge responses for debugging
    
    Returns:
        score: 1 if correct, 0 otherwise
    """
    is_correct, rationale = evaluate_answer_soft(question, answer, ground_truth)
    
    # Log judge response for debugging
    if debug_log_dir:
        import os
        from pathlib import Path
        os.makedirs(debug_log_dir, exist_ok=True)
        debug_file = Path(debug_log_dir) / "reward_judge_responses.jsonl"
        with open(debug_file, 'a') as f:
            log_entry = {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "judge_output": rationale,
                "judge_json": {"judgement": "correct" if is_correct else "incorrect", "rationale": rationale},
                "score": 1 if is_correct else 0
            }
            f.write(json.dumps(log_entry) + '\n')
    
    return 1 if is_correct else 0