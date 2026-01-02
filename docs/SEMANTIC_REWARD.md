# Semantic Reward Evaluation (Default)

## Overview

**Semantic evaluation is now the DEFAULT reward mechanism** for math tasks. This provides more lenient, semantically-aware evaluation that:
- Handles minor formatting differences
- Recognizes semantically equivalent answers
- Improves training stability and accuracy

The system automatically falls back to rule-based evaluation if semantic evaluation fails or isn't configured.

## Usage

### During Training (Default Behavior)

Just set your environment variables:

```bash
export CMU_GATEWAY_API_KEY="your-key"
export CMU_GATEWAY_BASE_URL="https://ai-gateway.andrew.cmu.edu"  # optional
export CMU_GATEWAY_MODEL="gpt-5-mini"  # optional
```

Semantic evaluation is now **enabled by default** - no extra config needed!

### Force Rule-Based Evaluation

To use strict rule-based evaluation instead:

```python
# Via extra_info
extra_info = {"use_semantic": False}

# Or via function call
compute_score(solution_str, ground_truth, use_semantic=False)
```

### Standalone Script

Use `reward_eval.py` to re-evaluate existing results:

```bash
python reward_eval.py --input_dir /path/to/results --fix_incorrect
```

## How It Works

1. **Semantic (Default)**: Uses LLM to check if predicted answer matches ground truth semantically
2. **Rule-based (Fallback)**: Extracts answer from `\boxed{}`, normalizes strings, exact match

The semantic prompt is simple and generic:
```
Check if the final answer matches the ground truth semantically.

Ground Truth: {ground_truth}
Predicted Answer: {predicted_answer}
```

## Configuration

- **CMU_GATEWAY_API_KEY**: Required for semantic evaluation (falls back gracefully if missing)
- **CMU_GATEWAY_BASE_URL**: Optional, defaults to CMU Gateway
- **CMU_GATEWAY_MODEL**: Optional, defaults to "gpt-5-mini"

If semantic evaluation fails, the system automatically falls back to rule-based evaluation.
