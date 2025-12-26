from .eval_kpr_async import evaluate_query_kpr
from .eval_quality_async import evaluate_query_quality
from .supergpqa_eval import evaluate_supergpqa_answer
from .afm_eval import evaluate_afm_answer
import sys
import os

def evaluation_reward_fn(query_id, question, answer, mode, ground_truth=None, options=None, debug_log_dir=None):
    """
    Evaluation reward function with optional debugging.
    
    Args:
        query_id: Question ID
        question: The question
        answer: The predicted answer
        mode: 'qa' or 'report'
        ground_truth: Ground truth answer (for QA mode)
        options: Options (for some datasets)
        debug_log_dir: Optional directory to save judge responses
    """
    if mode == 'report':
        kpr_result = evaluate_query_kpr(query_id, answer)
        quality_result = evaluate_query_quality(query_id, question, answer)
        combined_score = ((quality_result['normalized_score'] * 10 + kpr_result['support_rate']) / 2)
        return combined_score
    elif mode == "qa":
        score = evaluate_afm_answer(question, answer, ground_truth, debug_log_dir=debug_log_dir)
        return score            
