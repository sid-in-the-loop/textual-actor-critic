
import torch
import numpy as np
from verl import DataProto
from verl.utils.torch_functional import get_response_mask

# --- Turn 1: Initial Solution Template ---
TURN1_TEMPLATE = """Problem: {question}

Thinking Process:
1. Identify the core components of the problem.
2. Formulate a step-by-step plan to solve it.
3. Execute the steps clearly.
4. State the final answer in the required format. End with "The final answer is \\boxed{{answer}}".

Thinking Process:"""

# --- Turn 2: Pitfalls/Feedback Template ---
TURN2_TEMPLATE = """You are an expert math tutor. Review the following math problem and a student's initial attempt.
Identify the critical pitfalls, conceptual errors, or calculation mistakes. 
Provide a concise list of "pitfalls to avoid" or "hints" to solve the problem correctly.
Do NOT solve the problem. Use short, direct instructions (e.g., "Do X, avoid Y").

Problem: {question}
Student Attempt: {z_solution}
Pitfalls to Avoid:"""

# --- Turn 3: Refined Solution Template ---
TURN3_TEMPLATE = """Problem: {question}

Your Initial Attempt:
{z_solution}

Potential Suggestions for Improvement:
{g_feedback}

Thinking Process:
1. Identify the core components of the problem.
2. Formulate a step-by-step plan to solve it.
3. Execute the steps clearly.
4. State the final answer in the required format. End with "The final answer is \\boxed{{answer}}".

Thinking Process:"""

def prepare_mlmt_turn1_prompt(question):
    return TURN1_TEMPLATE.format(question=question)

def prepare_mlmt_feedback_prompt(question, z_solution):
    return TURN2_TEMPLATE.format(question=question, z_solution=z_solution)

def prepare_mlmt_refinement_prompt(question, z_solution, g_feedback):
    return TURN3_TEMPLATE.format(question=question, z_solution=z_solution, g_feedback=g_feedback)

# Code Templates
CODE_TURN1_TEMPLATE = """>>> Problem:
{question}

>>> Code:
"""

CODE_TURN2_TEMPLATE = """You are an expert programming mentor reviewing code written by a student.

>>> Problem:
{question}

>>> Student Solution:
{z_solution}

PROMPT: First, analyze the solution for bugs, inefficiencies, or edge cases it doesn't handle. Then, write a brief, helpful instruction that will guide the student toward correcting their solution. Your instruction should be specific to the issues you identified, but don't solve the problem completely for them. Your response should be ONLY the instruction for the student to improve their solution, nothing else. DO NOT write any code.

>>> Guiding Instruction:"""

CODE_TURN3_TEMPLATE = """>>> Problem:
{question}

>>> Initial Solution:
{z_solution}

>>> Code Review Feedback: 
{g_feedback}

Fix the issues above. Respond ONLY with the corrected code block in triple backticks. 
No other text is allowed.

>>> Refined Code:
"""

def prepare_mlmt_code_turn1_prompt(question):
    return CODE_TURN1_TEMPLATE.format(question=question)

def prepare_mlmt_code_feedback_prompt(question, z_solution, error=""):
    error_str = f"\n\nEXECUTION ERROR:\n{error}" if error else ""
    return CODE_TURN2_TEMPLATE.format(question=question, z_solution=z_solution) + error_str

def prepare_mlmt_code_refinement_prompt(question, z_solution, g_feedback):
    return CODE_TURN3_TEMPLATE.format(question=question, z_solution=z_solution, g_feedback=g_feedback)

def compute_mlmt_bi_level_reward(task_reward, vl_x_g, vl_star_x_g=0.0, lambda_coef=0.1):
    return task_reward + lambda_coef * (vl_x_g - vl_star_x_g)

def apply_symmetric_reaping(rewards):
    if isinstance(rewards, torch.Tensor):
        return torch.where(rewards > 0, torch.ones_like(rewards), -torch.ones_like(rewards))
    return np.where(rewards > 0, 1.0, 0.0)

# --- SCoRe Prompt Templates ---
SCORE_TURN1_TEMPLATE = """Problem: {question}

Thinking Process:
1. Identify the core components of the problem.
2. Formulate a step-by-step plan to solve it.
3. Execute the steps clearly.
4. State the final answer in the required format. End with "The final answer is \\boxed{{answer}}".

Thinking Process:"""

SCORE_TURN2_INSTRUCTION = """There might be an error in the initial attempt above.
Please carefully re-evaluate the problem, identify any mistakes, and provide a corrected step-by-step solution.

Thinking Process:
1. Identify the core components of the problem.
2. Formulate a step-by-step plan to solve it.
3. Execute the steps clearly.
4. State the final answer in the required format. End with "The final answer is \\boxed{{answer}}".

Thinking Process:"""


def build_score_turn1_prompt(question: str, template: str | None = None) -> str:
    """Construct the stage-agnostic first-turn prompt for SCoRe."""
    tmpl = template or SCORE_TURN1_TEMPLATE
    return tmpl.format(question=question.strip())


def build_score_turn2_prompt(question: str, turn1_solution: str, instruction: str | None = None) -> str:
    """Construct the self-correction turn prompt given the question and turn-1 text."""
    extra_instruction = instruction or SCORE_TURN2_INSTRUCTION
    return (
        f"Problem: {question.strip()}\n\n"
        f"Initial Attempt:\n{turn1_solution.strip()}\n\n"
        f"{extra_instruction.strip()}\n"
    )
