
import torch
import numpy as np
import re
from verl import DataProto
from verl.utils.torch_functional import get_response_mask

# --- Extraction Utility ---
def extract_math_final_answer(text: str) -> str:
    """Extract the final answer from a math solution using LaTeX box or 'The final answer is'."""
    # 1. Search for \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 2. Search for "The final answer is: ...", "The answer is: ...", "Answer: ...", or "Final Answer: ..."
    ans_match = re.search(r'(?:The final answer is|The answer is|Final Answer|Answer):\s*(.*)', text, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip()
    
    # 3. Last fallback: look for the last line if it seems like a short answer
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines and len(lines[-1]) < 100:
        return lines[-1]
        
    return "unsolved"

# --- Turn 1: Initial Solution Template ---
TURN1_TEMPLATE = """Problem: {question}

Thinking Process:
1. Identify the core components of the problem.
2. Formulate a step-by-step plan to solve it.
3. Execute the steps clearly.
4. State the final answer in the required format.

Thinking Process:"""

# --- Turn 2: Pitfalls/Feedback Template ---
TURN2_TEMPLATE = """You are an expert math tutor. Review the following math problem and a student's initial attempt.
Deeply analyze where the model went wrong if it did and provide specific, actionable feedback on how to improve that.
Identify critical pitfalls, conceptual errors, or calculation mistakes.
Provide a concise list of "pitfalls to avoid" or "hints" to solve the problem correctly.
Do NOT solve the problem. Use short, direct instructions.

Problem: {question}
Student Attempt: {z_solution}
Feedback:"""

# --- Turn 3: Refined Solution Template ---
TURN3_TEMPLATE = """Problem: {question}

Your Previous Final Answer: 
{final_answer}

Feedback for Improvement:
{g_feedback}

Thinking Process:
1. Analyze the received feedback and identify where the previous attempt went wrong.
2. Identify the core components of the problem.
3. Think through step-by-step and solve the problem correctly.
5. State the final answer in the required format: "Answer: $Answer".

Thinking Process:
Solve the following math problem step by step."""

def prepare_mlmt_turn1_prompt(question):
    return TURN1_TEMPLATE.format(question=question)

def prepare_mlmt_feedback_prompt(question, z_solution):
    return TURN2_TEMPLATE.format(question=question, z_solution=z_solution)

def prepare_mlmt_refinement_prompt(question, final_answer, g_feedback):
    return TURN3_TEMPLATE.format(question=question, final_answer=final_answer, g_feedback=g_feedback)

# Code Templates
CODE_TURN1_TEMPLATE = """You are an expert programmer. Below is a programming problem. Write a solution in python.
Make sure your solution is correct, efficient, and addresses all the requirements of the problem.
When you're done, wrap your code in triple backticks with the language specified, like: ```python (your code here) ```

Problem:
{question}

Solution:
"""

CODE_TURN2_TEMPLATE = """You are an expert programming mentor reviewing code written by a student.

PROBLEM:
{question}

STUDENT'S SOLUTION:
{z_solution}

PROMPT: First, analyze the solution for bugs, inefficiencies, or edge cases it doesn't handle. Then, write a brief, helpful instruction that will guide the student toward correcting their solution. Your instruction should be specific to the issues you identified, but don't solve the problem completely for them. Your response should be ONLY the instruction for the student to improve their solution, nothing else. DO NOT write any code.

GUIDING INSTRUCTION:"""

CODE_TURN3_TEMPLATE = """Problem:
{question}

Initial Solution:
{z_solution}

Code Review Feedback: 
{g_feedback}

Please fix these issues and provide an improved solution. Remember to wrap your code in triple backticks with the language specified, like: ```python (your code here) ```
"""

def prepare_mlmt_code_turn1_prompt(question):
    return CODE_TURN1_TEMPLATE.format(question=question)

def prepare_mlmt_code_feedback_prompt(question, z_solution):
    return CODE_TURN2_TEMPLATE.format(question=question, z_solution=z_solution)

def prepare_mlmt_code_refinement_prompt(question, z_solution, g_feedback):
    return CODE_TURN3_TEMPLATE.format(question=question, z_solution=z_solution, g_feedback=g_feedback)

def compute_mlmt_bi_level_reward(task_reward, vl_x_g, vl_star_x_g=0.0, lambda_coef=0.1):
    return task_reward + lambda_coef * (vl_x_g - vl_star_x_g)

def apply_symmetric_reaping(rewards):
    if isinstance(rewards, torch.Tensor):
        return torch.where(rewards > 0, torch.ones_like(rewards), -torch.ones_like(rewards))
    return np.where(rewards > 0, 1.0, 0.0)

# --- SCoRe Prompt Templates ---
SCORE_TURN1_TEMPLATE = """Solve the following math problem step by step.

Problem:
{question}

Solution:
"""

SCORE_TURN2_INSTRUCTION = """There might be an error in the solution above due to a misunderstanding of the problem.
Please carefully check the reasoning, correct any mistakes if they exist,
and rewrite the full solution."""


def build_score_turn1_prompt(question: str, template: str | None = None) -> str:
    """Construct the stage-agnostic first-turn prompt for SCoRe."""
    tmpl = template or SCORE_TURN1_TEMPLATE
    return tmpl.format(question=question.strip())


def build_score_turn2_prompt(question: str, turn1_solution: str, instruction: str | None = None) -> str:
    """Construct the self-correction turn prompt given the question and turn-1 text."""
    extra_instruction = instruction or SCORE_TURN2_INSTRUCTION
    return (
        f"Problem:\n{question.strip()}\n\n"
        f"Solution:\n{turn1_solution.strip()}\n\n"
        f"{extra_instruction.strip()}\n"
    )