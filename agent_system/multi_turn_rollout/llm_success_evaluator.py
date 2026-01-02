import asyncio
import json
import os
from typing import Dict, List

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None


class LLMSuccessEvaluator:
    """Lightweight async evaluator that scores final answers using an LLM."""

    def __init__(self, config=None):
        config = config or {}
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.0)
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key) if (AsyncOpenAI and api_key) else None

    async def _evaluate_single(self, question: str, answer: str) -> Dict[str, float]:
        if not self.client:
            return self._heuristic_score(question, answer)

        prompt = (
            "You are an impartial evaluator for math and reasoning problems.\n"
            "Given the question and the model's final answer, respond with a JSON object \
            containing two fields: 'success' (0.0-1.0 indicating correctness) and \
            'feedback_quality' (0.0-1.0 describing how clear/confident the answer is).\n"
            "JSON only."
        )
        content = f"Question: {question}\nAnswer: {answer}"
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
            )
            payload = response.choices[0].message.content.strip()
            parsed = json.loads(payload)
            success = float(parsed.get("success", 0.0))
            feedback_quality = float(parsed.get("feedback_quality", success))
            return {"success": success, "feedback_quality": feedback_quality}
        except Exception as exc:  # pragma: no cover
            print(f"LLM success evaluator failed: {exc}")
            return self._heuristic_score(question, answer)

    async def _evaluate_batch_async(self, questions: List[str], answers: List[str]):
        tasks = [self._evaluate_single(q, a) for q, a in zip(questions, answers)]
        return await asyncio.gather(*tasks)

    def evaluate_batch(self, questions: List[str], answers: List[str]):
        if not questions:
            return []
        try:
            return asyncio.run(self._evaluate_batch_async(questions, answers))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self._evaluate_batch_async(questions, answers))
            loop.close()
            return results

    @staticmethod
    def _heuristic_score(question: str, answer: str) -> Dict[str, float]:
        if not answer:
            return {"success": 0.0, "feedback_quality": 0.0}
        answer = answer.lower()
        success = 1.0 if "final answer" in answer else 0.3
        feedback_quality = 0.5 if len(answer) > 20 else 0.2
        return {"success": success, "feedback_quality": feedback_quality}
