# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Belief Calculator for Belief-Shaped GRPO

Integrates belief-based value function computation into the rollout process.
Computes smoothed belief distributions and value functions during trajectory collection.

ULTRA OPTIMIZED VERSION:
- Simplified prompts for reduced latency (~60-70% fewer tokens)
- max_candidates reduced to 1 (single hypothesis extraction)
- ENTIRELY REMOVED ENTAILMENT CHECKS (75% API call reduction)
- Belief = concentration only (add/subtract handles correctness)
- Thread-safe caching with MD5 hashing
"""

from openai import OpenAI
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class BeliefCalculator:
    """
    Computes belief-based value functions for shaping GRPO rewards.

    This integrates the belief computation logic from is_checkpoint4.py
    into the rollout process for real-time belief score calculation.
    """

    def __init__(self, alpha: float = 1.0, max_candidates: int = 2, max_workers: int = 1):
        """
        Initialize the belief calculator.

        Args:
            alpha: Smoothing parameter for belief distribution (default: 1.0)
            max_candidates: Maximum number of hypotheses to extract (default: 3)
            max_workers: Maximum concurrent threads for API requests (default: 1)
        """
        self.alpha = alpha
        self.max_candidates = max_candidates
        self.max_workers = max_workers

        # Initialize OpenAI client using standard OpenAI API
        api_key = os.getenv("OPENAI_API_KEY", "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")
        self.client = OpenAI(
            api_key=api_key,
        )

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="belief-api")
        self._entailment_cache = {}
        self._belief_cache = {}  # Cache for full belief computations
        self._cache_lock = threading.Lock()  # Thread-safe cache access

        print(f"🔧 BeliefCalculator initialized with {max_workers} concurrent workers")
        print(f"⚡ ULTRA Optimizations: max_candidates={max_candidates}, NO ENTAILMENT, concentration-only belief")

    def extract_candidate_hypotheses_with_support(self, evidence_docs: List[str], question: str) -> List[Dict]:
        """
        Extract candidate answer hypotheses with evidence support counts and snippets.
        OPTIMIZED: Simplified prompt with ~60% fewer tokens.
        """
        numbered_evidence = []
        for i, doc in enumerate(evidence_docs):
            numbered_evidence.append(f"[Doc {i+1}] {doc}")

        combined_evidence = "\n\n".join(numbered_evidence)

        # OPTIMIZED PROMPT: Reduced from ~400 tokens to ~150 tokens
        prompt = f"""Extract candidate answers from evidence.

Question: {question}

Evidence:
{combined_evidence}

Rules:
- Extract answers matching question type (year→year, %→%, name→name)
- Max {self.max_candidates} answers
- Return NO_ANSWER if none found
- Paraphrases of same value = same hypothesis

Format each:
{{"hypothesis": "answer", "count": supporting_docs, "snippet": "1-2 sentences"}}

Return JSON array only.
Example: [{{"hypothesis": "2015", "count": 3, "snippet": "Team promoted in 2015..."}}, {{"hypothesis": "1993", "count": 1, "snippet": "Restructured in 1993..."}}]
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0
            )

            content = response.choices[0].message.content.strip()

            try:
                hypotheses = json.loads(content)
                if isinstance(hypotheses, list):
                    result = []
                    for h in hypotheses[:self.max_candidates]:
                        if isinstance(h, dict) and "hypothesis" in h:
                            hyp = h["hypothesis"].strip()
                            count = int(h.get("count", 0))
                            snippet = h.get("snippet", "").strip()
                            if hyp:
                                result.append({"hypothesis": hyp, "count": count, "snippet": snippet})
                    if result:
                        return result
            except json.JSONDecodeError:
                pass

            # Fallback: extract JSON from text
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    hypotheses = json.loads(json_match.group(0))
                    if isinstance(hypotheses, list):
                        result = []
                        for h in hypotheses[:self.max_candidates]:
                            if isinstance(h, dict) and "hypothesis" in h:
                                hyp = h["hypothesis"].strip()
                                count = int(h.get("count", 1))
                                snippet = h.get("snippet", "").strip()
                                if hyp:
                                    result.append({"hypothesis": hyp, "count": count, "snippet": snippet})
                        if result:
                            return result
                except:
                    pass

            return [{"hypothesis": "NO_ANSWER", "count": 0, "snippet": ""}]

        except Exception as e:
            print(f"Error extracting hypotheses: {e}")
            return [{"hypothesis": "NO_ANSWER", "count": 0, "snippet": ""}]

    def normalize_hypothesis(self, hypothesis: str) -> str:
        """Normalize hypothesis for grouping (handle dates, numbers, etc.)."""
        hyp = hypothesis.lower().strip()

        # Normalize years (4-digit numbers)
        year_match = re.search(r'\b(19|20)\d{2}\b', hyp)
        if year_match:
            return year_match.group(0)

        # Normalize percentages
        pct_match = re.search(r'\b(\d+(?:\.\d+)?)%\b', hyp)
        if pct_match:
            return f"{pct_match.group(1)}%"

        # Normalize currency
        currency_match = re.search(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', hyp)
        if currency_match:
            return f"${currency_match.group(1)}"

        return hyp

    def build_normalized_belief(self, hypotheses_with_support: List[Dict]) -> List[Dict]:
        """
        Build smoothed belief distribution with explicit OTHER hypothesis.

        Returns: List of hypothesis dicts with keys:
            - hypothesis: original text
            - normalized: normalized form
            - support_count: raw count
            - probability: smoothed probability
            - snippet: supporting evidence
        """
        if not hypotheses_with_support:
            return [{
                "hypothesis": "OTHER",
                "normalized": "OTHER",
                "support_count": 0,
                "probability": 1.0,
                "snippet": ""
            }]

        # Filter out NO_ANSWER
        valid_hypotheses = [h for h in hypotheses_with_support if h.get("hypothesis", "") != "NO_ANSWER"]

        if not valid_hypotheses:
            return [{
                "hypothesis": "OTHER",
                "normalized": "OTHER",
                "support_count": 0,
                "probability": 1.0,
                "snippet": ""
            }]

        # Aggregate by normalized form
        agg_counts = defaultdict(int)
        agg_snippets = {}
        agg_original = {}

        for h in valid_hypotheses:
            hyp_original = h["hypothesis"]
            hyp_normalized = self.normalize_hypothesis(hyp_original)
            count = h.get("count", 1)
            snippet = h.get("snippet", "")

            agg_counts[hyp_normalized] += count

            # Keep first occurrence for snippet and original
            if hyp_normalized not in agg_snippets:
                agg_snippets[hyp_normalized] = snippet
                agg_original[hyp_normalized] = hyp_original

        # Compute smoothed probabilities
        K = len(agg_counts)  # Number of extracted hypotheses
        total_count = sum(agg_counts.values())
        denominator = total_count + self.alpha * (K + 1)

        items = []
        for norm_hyp, count in agg_counts.items():
            prob = (count + self.alpha) / denominator
            items.append({
                "hypothesis": agg_original[norm_hyp],
                "normalized": norm_hyp,
                "support_count": count,
                "probability": prob,
                "snippet": agg_snippets[norm_hyp]
            })

        # Add OTHER hypothesis
        prob_other = self.alpha / denominator
        items.append({
            "hypothesis": "OTHER",
            "normalized": "OTHER",
            "support_count": 0,
            "probability": prob_other,
            "snippet": ""
        })

        # Sort by probability (descending)
        items.sort(key=lambda x: x["probability"], reverse=True)

        return items

    def compute_belief_concentration(self, normalized_belief: List[Dict]) -> float:
        """
        Compute concentration from normalized belief distribution.
        Returns the top probability (no gap heuristic).
        """
        if not normalized_belief:
            return 0.0

        return normalized_belief[0]["probability"]

    def check_entailment_support(self, question: str, hypothesis_original: str, supporting_snippet: str) -> float:
        """
        Check if the supporting snippet entails the hypothesis as the answer.
        OPTIMIZED: Skip LLM call if hypothesis appears verbatim in snippet.
        """
        if hypothesis_original in ["NO_ANSWER", "OTHER"] or not supporting_snippet:
            return 0.0

        # OPTIMIZATION: Fast path for verbatim matches
        # If hypothesis appears in snippet (case-insensitive), assume entailment
        if hypothesis_original.lower() in supporting_snippet.lower():
            return 1.0

        # Check cache - use hash of snippet to avoid truncation collisions
        snippet_hash = hashlib.md5(supporting_snippet.encode()).hexdigest()
        cache_key = f"{question}|||{hypothesis_original}|||{snippet_hash}"
        with self._cache_lock:
            if cache_key in self._entailment_cache:
                return self._entailment_cache[cache_key]

        # OPTIMIZED PROMPT: Reduced from ~250 tokens to ~80 tokens
        prompt = f"""Does the snippet explicitly answer the question with this hypothesis?

Question: {question}
Hypothesis: {hypothesis_original}
Snippet: {supporting_snippet}

Score:
1.0 = snippet clearly states this answer
0.0 = snippet does not state this answer

Return only: 1.0 or 0.0"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0
            )

            content = response.choices[0].message.content.strip()
            number_match = re.search(r'(\d+(?:\.\d+)?)', content)
            if number_match:
                score = float(number_match.group(1))
                score = max(0.0, min(1.0, score))
                with self._cache_lock:
                    self._entailment_cache[cache_key] = score
                return score

        except Exception as e:
            print(f"Error checking entailment: {e}")

        with self._cache_lock:
            self._entailment_cache[cache_key] = 0.0
        return 0.0

    def belief_based_value_function(self, evidence_docs: List[str], question: str,
                                   previous_belief: Optional[float] = None,
                                   temporal_alpha: float = 0.3) -> Tuple[float, Dict]:
        """
        Compute belief-based value function.

        V_t = p_top * s(h_top)

        Returns: (value: float, metadata: dict)
        """
        if not evidence_docs:
            return 0.0, {
                "normalized_belief": [],
                "concentration": 0.0,
                "support": 0.0,
                "top_hypothesis": "OTHER",
                "top_hypothesis_original": "OTHER",
                "snippet": "",
                "raw_hypotheses": []
            }

        # Extract candidate hypotheses
        raw_hypotheses = self.extract_candidate_hypotheses_with_support(evidence_docs, question)

        # Build smoothed belief distribution with OTHER
        normalized_belief = self.build_normalized_belief(raw_hypotheses)

        # Compute concentration (top probability) - THIS IS NOW OUR BELIEF SCORE
        concentration = self.compute_belief_concentration(normalized_belief)
        top_item = normalized_belief[0] if normalized_belief else {"hypothesis": "OTHER", "normalized": "OTHER", "snippet": ""}

        # OPTIMIZATION: Remove entailment entirely - concentration alone suffices
        # with add/subtract mechanism for correctness handling
        support = 1.0 if top_item["normalized"] != "OTHER" else 0.0

        # Compute value: V = concentration (entailment removed for speed)
        raw_value = concentration

        # Optional: temporal smoothing
        if previous_belief is not None:
            value = temporal_alpha * previous_belief + (1 - temporal_alpha) * raw_value
        else:
            value = raw_value

        metadata = {
            "normalized_belief": normalized_belief,
            "concentration": concentration,
            "support": support,
            "top_hypothesis": top_item["normalized"],
            "top_hypothesis_original": top_item["hypothesis"],
            "snippet": top_item["snippet"],
            "raw_hypotheses": raw_hypotheses
        }

        return value, metadata

    def compute_belief_scores_batch(self, batch_data: List[Tuple[str, str, Optional[float]]]) -> List[Tuple[float, Dict]]:
        """
        Compute belief scores for a batch of (k_t_combined, question, previous_belief) tuples.
        Uses ThreadPoolExecutor for concurrent processing without asyncio conflicts.

        Args:
            batch_data: List of (k_t_combined, question, previous_belief) tuples

        Returns:
            List of (belief_score, metadata) tuples
        """
        # Submit all tasks to thread pool
        future_to_index = {}
        for i, (k_t, q, prev) in enumerate(batch_data):
            future = self._executor.submit(self._compute_belief_score_sync, k_t, q, prev)
            future_to_index[future] = i

        # Collect results as they complete
        results = [None] * len(batch_data)
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Error in concurrent belief computation for item {index}: {e}")
                results[index] = (0.0, {})

        return results

    def _compute_belief_score_sync(self, k_t_combined: str, question: str, previous_belief: Optional[float] = None) -> Tuple[float, Dict]:
        """
        Synchronous version of belief computation to avoid asyncio event loop issues.
        """
        if not k_t_combined or not k_t_combined.strip():
            return 0.0, {}

        # Split evidence into individual documents
        evidence_docs = [d.strip() for d in k_t_combined.split('\n\n') if d.strip()]

        if not evidence_docs:
            return 0.0, {}

        # Extract candidate hypotheses (synchronous)
        raw_hypotheses = self._extract_candidate_hypotheses_sync(evidence_docs, question)

        # Build smoothed belief distribution with OTHER
        normalized_belief = self.build_normalized_belief(raw_hypotheses)

        # Compute concentration (top probability)
        concentration = self.compute_belief_concentration(normalized_belief)
        top_item = normalized_belief[0] if normalized_belief else {"hypothesis": "OTHER", "normalized": "OTHER", "snippet": ""}

        # OPTIMIZATION: Remove entailment entirely - concentration alone suffices
        # with add/subtract mechanism for correctness handling
        support = 1.0 if top_item["normalized"] != "OTHER" else 0.0

        # Compute value: V = concentration (entailment removed for speed)
        raw_value = concentration

        temporal_alpha = 0.25
        if previous_belief is not None:
            value = temporal_alpha * previous_belief + (1 - temporal_alpha) * raw_value
        else:
            value = raw_value

        metadata = {
            "normalized_belief": normalized_belief,
            "concentration": concentration,
            "support": support,
            "top_hypothesis": top_item["normalized"],
            "top_hypothesis_original": top_item["hypothesis"],
            "snippet": top_item["snippet"],
            "raw_hypotheses": raw_hypotheses
        }

        return value, metadata

    def _extract_candidate_hypotheses_sync(self, evidence_docs: List[str], question: str) -> List[Dict]:
        """
        Synchronous version of hypothesis extraction.
        OPTIMIZED: Simplified prompt with ~60% fewer tokens.
        """
        # Format evidence consistently with main method
        numbered_evidence = [f"[Doc {i+1}] {doc}" for i, doc in enumerate(evidence_docs)]
        evidence_text = "\n\n".join(numbered_evidence)

        # OPTIMIZED PROMPT: Reduced from ~400 tokens to ~150 tokens
        prompt = f"""Extract candidate answers from evidence.

Question: {question}

Evidence:
{evidence_text}

Rules:
- Extract answers matching question type (year→year, %→%, name→name)
- Max {self.max_candidates} answers
- Return NO_ANSWER if none found
- Paraphrases of same value = same hypothesis

Format each:
{{"hypothesis": "answer", "count": supporting_docs, "snippet": "1-2 sentences"}}

Return JSON array only.
Example: [{{"hypothesis": "2015", "count": 3, "snippet": "Team promoted in 2015..."}}, {{"hypothesis": "1993", "count": 1, "snippet": "Restructured in 1993..."}}]
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0
            )

            content = response.choices[0].message.content.strip()

            try:
                hypotheses = json.loads(content)
                if isinstance(hypotheses, list):
                    result = []
                    for h in hypotheses[:self.max_candidates]:
                        if isinstance(h, dict) and "hypothesis" in h:
                            hyp = h["hypothesis"].strip()
                            count = int(h.get("count", 0))
                            snippet = h.get("snippet", "").strip()
                            if hyp:
                                result.append({
                                    "hypothesis": hyp,
                                    "count": count,
                                    "snippet": snippet
                                })
                    return result
            except json.JSONDecodeError:
                pass

        except Exception as e:
            print(f"Error in sync hypothesis extraction: {e}")

        return [{"hypothesis": "NO_ANSWER", "count": 0, "snippet": ""}]

    def _check_entailment_support_sync(self, question: str, hypothesis_original: str, supporting_snippet: str) -> float:
        """
        Synchronous version of entailment checking.
        OPTIMIZED: Skip LLM call if hypothesis appears verbatim in snippet.
        """
        if hypothesis_original in ["NO_ANSWER", "OTHER"] or not supporting_snippet:
            return 0.0

        # OPTIMIZATION: Fast path for verbatim matches
        # If hypothesis appears in snippet (case-insensitive), assume entailment
        if hypothesis_original.lower() in supporting_snippet.lower():
            return 1.0

        # Check cache - use hash of snippet to avoid truncation collisions
        snippet_hash = hashlib.md5(supporting_snippet.encode()).hexdigest()
        cache_key = f"{question}|||{hypothesis_original}|||{snippet_hash}"
        with self._cache_lock:
            if cache_key in self._entailment_cache:
                return self._entailment_cache[cache_key]

        # OPTIMIZED PROMPT: Reduced from ~250 tokens to ~80 tokens
        prompt = f"""Does the snippet explicitly answer the question with this hypothesis?

Question: {question}
Hypothesis: {hypothesis_original}
Snippet: {supporting_snippet}

Score:
1.0 = snippet clearly states this answer
0.0 = snippet does not state this answer

Return only: 1.0 or 0.0"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0
            )

            content = response.choices[0].message.content.strip()
            number_match = re.search(r'(\d+(?:\.\d+)?)', content)
            if number_match:
                score = float(number_match.group(1))
                score = max(0.0, min(1.0, score))
                with self._cache_lock:
                    self._entailment_cache[cache_key] = score
                return score

        except Exception as e:
            print(f"Error in sync entailment check: {e}")

        return 0.0

    def compute_belief_score(self, k_t_combined: str, question: str, previous_belief: Optional[float] = None) -> Tuple[float, Dict]:
        """
        Compute belief score for a single turn of evidence.

        Args:
            k_t_combined: Combined evidence string (K_t_combined)
            question: The question being answered
            previous_belief: Previous belief score for temporal smoothing

        Returns:
            Tuple of (belief_score, metadata)
        """
        # Check cache first (thread-safe cache for identical inputs)
        cache_key = hashlib.md5(f"{k_t_combined}|{question}|{previous_belief}".encode()).hexdigest()
        with self._cache_lock:
            if cache_key in self._belief_cache:
                return self._belief_cache[cache_key]

        # For single computations, use the batch method with one item
        results = self.compute_belief_scores_batch([(k_t_combined, question, previous_belief)])
        result = results[0]

        # Cache the result (thread-safe)
        with self._cache_lock:
            self._belief_cache[cache_key] = result

        return result

    def shutdown(self):
        """Shutdown the thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)