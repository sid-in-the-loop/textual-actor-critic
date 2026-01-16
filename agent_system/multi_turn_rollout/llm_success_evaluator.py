import os
import json
import time
from typing import List, Dict, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

PROMPT_TEMPLATE = """You are an expert math grader. Your task is to determine if a student's answer is mathematically equivalent to the ground truth answer.

Guidelines:
1. Focus on mathematical equivalence, not formatting.
2. If the ground truth is a fraction like 1/2, then 0.5 is correct.
3. If the ground truth is a simplified expression, the student's answer must be equivalent.
4. Be strict but fair.

Problem: {question}
Ground Truth: {ground_truth}
Predicted Answer: {answer}

Please respond with a JSON object:
{{
"judgement": "correct" or "incorrect",
"rationale": "short explanation"
}}
"""

class LLMSuccessEvaluator:
    """Simplified evaluator that scores answers using OpenAI API with threading."""

    def __init__(self, config=None):
        config = config or {}
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.1)
        # Use provided key as fallback to ensure Ray workers can always authenticate
        env_key = os.getenv("OPENAI_API_KEY")
        self.api_key = env_key 
        # Initialize client
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.workers = config.get("workers", 32)

    def _evaluate_single(self, question: str, answer: str, ground_truth: str) -> float:
        """Evaluate a single answer against ground truth."""
        if not self.client or not ground_truth:
            # Fallback to rule-based math verifier
            try:
                from verl.utils.reward_score.math import compute_score
                return float(compute_score(answer, ground_truth))
            except:
                return 0.0

        prompt = PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            answer=answer
        )

        # Simple retry loop for API robustness
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    timeout=20.0
                )
                payload = response.choices[0].message.content.strip()
                res = json.loads(payload)
                judgement = res.get('judgement', '').lower()
                success = 1.0 if judgement == 'correct' else 0.0

                # --- NEW LOGGING ---
                try:
                    log_dir = "logs/judgments"
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, f"judgments_{time.strftime('%Y%m%d')}.jsonl")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "timestamp": time.strftime("%H:%M:%S"),
                            "question": question[:200],
                            "ground_truth": ground_truth,
                            "answer": answer[:500],
                            "rationale": res.get("rationale", ""),
                            "judgement": judgement,
                            "success": success
                        }) + "\n")
                except:
                    pass

                return success
            except Exception as e:
                if attempt == 1:
                    print(f"⚠️ [LLM Evaluator] API Call failed: {e}")
                time.sleep(1)

        return 0.0

    def evaluate_batch(self, questions: List[str], answers: List[str], ground_truths: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """Evaluates a batch of answers in parallel."""
        if ground_truths is None:
            return [{"success": 0.0} for _ in range(len(questions))]

        # Ensure all are strings
        questions = [str(q) for q in questions]
        answers = [str(a) for a in answers]
        ground_truths = [str(gt) for gt in ground_truths]

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(self._evaluate_single, q, a, gt)
                for q, a, gt in zip(questions, answers, ground_truths)
            ]
            results = [f.result() for f in futures]

        return [{"success": res} for res in results]
