# Textual Actor-Critic Beyond Training

This repository implements **Textual Actor-Critic Beyond Training**, a hierarchical framework for training Large Language Models (LLMs) to reason more effectively by generating and utilizing high-quality in-context demonstrations and feedback.

---

## 🚀 Getting Started

### 1. Environment Setup

We recommend using **Conda** for environment management. This project requires **Python 3.12**.

```bash
# Create a new conda environment
conda create -n mlmt python=3.12 -y
conda activate mlmt

# Install dependencies (pinned for stability)
pip install -r requirements.txt

# Install the project (verl) in editable mode
pip install -e .
```

---

## Method Theory: MLMT-RL

MLMT-RL is based on a hierarchical interaction between two levels of models:

1.  **High-Level (HL) - Context Generator $\pi_\theta(c \mid x)$**:
    *   Generates synthetic demonstrations or feedback $c$ for a given problem $x$.
    *   Trained to maximize the success and confidence of the Lower-Level model.

2.  **Low-Level (LL) - Task Solver $p_\phi(y \mid x, c)$**:
    *   Solves the task $x$ conditioned on the context $c$ provided by the HL.
    *   Trained via **Group Relative Policy Optimization (GRPO)**.

### The Three-Turn Loop
The training process follows a structured interaction:
1.  **Turn 1 (Initial Solve)**: LL samples an initial solution $z \sim \pi_L(\cdot \mid x)$.
2.  **Turn 2 (Feedback)**: HL samples feedback or context $g \sim \pi_H(\cdot \mid x, z)$.
3.  **Turn 3 (Refinement)**: LL samples a final refined solution $\hat{y} \sim \pi_L(\cdot \mid x, z, g)$.

---

## 📂 Project Structure & Scripts

### Configuration & Launch Scripts
Configurations and launch scripts are organized by task and level of training.

#### Math Reasoning (`mlmt-configs/math/`)
These scripts are used for training on mathematical reasoning tasks:
*   `HLT_LLT_RR_1_1.sh`: HL and LL both trainable
*   `HLT_LLT_RR_1_15.sh`: HL and LL both trainable
*   `HLT_LLT_RR_1_20.sh`: HL and LL both trainable
*   `submit_mlmt.sbatch`: Slurm submission script for launching MLMT training on a cluster.


### Core Modules
*   `agent_system/multi_turn_rollout/rollout_loop.py`: Implements the `mlmt_multi_turn_loop` logic.
*   `verl/trainer/ppo/ray_trainer.py`: Main trainer class using Ray for distributed compute.

---

## 🛠 Usage

### Training on Slurm
To launch a specific experiment on a Slurm cluster:
```bash
sbatch mlmt-configs/math/submit_mlmt.sbatch mlmt-configs/math/HLT_LLT_RR_1_1.sh
```

### Evaluation
To evaluate a trained model on the MATH-500 dataset:
```bash
./mlmt-configs/math/eval_math500.sh <model_path>
```

---

## 📊 Logging & Reproducibility
We use **Weights & Biases (W&B)** for experiment tracking. Ensure you are logged in:
```bash
wandb login
```
All training runs, including rewards and KL divergence, are logged under the project `mlmt_rl`.
