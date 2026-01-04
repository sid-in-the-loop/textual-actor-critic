
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
4. State the final answer in the required format.

Example:
Problem: What is 2+2?
Thinking Process:
1. Core component: Addition of two single-digit integers.
2. Plan: Add 2 and 2.
3. Execution: 2 + 2 = 4.
4. Final Answer: The final answer is $4$. I hope it is correct.

Now, solve the following problem:
Problem: {question}
Thinking Process:"""

# --- Turn 2: Pitfalls/Feedback Template ---
TURN2_TEMPLATE = """You are an expert math tutor. Review the following math problem and a student's initial attempt.
Identify the critical pitfalls, conceptual errors, or calculation mistakes. 
Provide a concise list of "pitfalls to avoid" or "hints" to solve the problem correctly.
Do NOT solve the problem. Use short, direct instructions (e.g., "Do X, avoid Y").

Example:
Problem: Find the area of a circle with radius 5.
Student Attempt: Area = 2 * pi * r = 10pi.
Pitfalls to Avoid:
- Don't confuse the circumference formula (2*pi*r) with the area formula (pi*r^2).
- Ensure you square the radius before multiplying by pi.

Problem: {question}
Student Attempt: {z_solution}
Pitfalls to Avoid:"""

# --- Turn 3: Refined Solution Template ---
TURN3_TEMPLATE = """Problem: {question}

Thinking Process:
Think deeply about the problem and pay close attention to the following pitfalls. Solve the question step-by-step from scratch.

PITFALLS TO AVOID:
{g_feedback}

Example:
Problem: Find the area of a circle with radius 5.
PITFALLS TO AVOID:
- Don't confuse the circumference formula (2*pi*r) with the area formula (pi*r^2).
- Ensure you square the radius before multiplying by pi.
Thinking Process:
1. Formula: Area = pi * r^2.
2. Substitution: Area = pi * (5^2).
3. Calculation: Area = 25 * pi.
4. Final Answer: The final answer is $25\\pi$. I hope it is correct.

Now, solve the following problem:
Problem: {question}
PITFALLS TO AVOID:
{g_feedback}
Thinking Process:"""

def prepare_mlmt_turn1_prompt(question):
    return TURN1_TEMPLATE.format(question=question)

def prepare_mlmt_feedback_prompt(question, z_solution):
    return TURN2_TEMPLATE.format(question=question, z_solution=z_solution)

def prepare_mlmt_refinement_prompt(question, g_feedback):
    return TURN3_TEMPLATE.format(question=question, g_feedback=g_feedback)

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
