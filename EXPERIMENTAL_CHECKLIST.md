# Hierarchical In-Context RL Experimental Checklist

## Phase-wise Experiment Plan

### **PHASE 0 — Baseline sanity**

- [ ] **Exp 1**: GRPO baseline
  - LL trained with GRPO
  - No context anywhere
  - Eval: MATH500
  - **Purpose**: establish baseline accuracy + variance

---

### **PHASE 1 — Context helps or not (no forcing)**

*(pure GRPO, exactly your current implementation)*

- [ ] **Exp 2**: Pretrained HL → test-time context only
  - HL pretrained
  - LL frozen
  - Context only at inference
  - **Purpose**: ICL-style effect without training mismatch

- [ ] **Exp 3**: Pretrained HL → training-time context
  - HL pretrained
  - LL trained with GRPO + context
  - Eval with context
  - **Purpose**: does GRPO benefit from *training-time* context?

- [ ] **Exp 4**: SFT HL → test-time context only
  - HL SFT on DSFT
  - LL frozen
  - **Purpose**: isolate HL quality at inference

- [ ] **Exp 5**: SFT HL → training-time context
  - HL SFT
  - LL trained with GRPO + context
  - Eval with context
  - **Purpose**: best "no-forcing" setup; strong baseline before KL

---

### **PHASE 2 — Does the LL actually *use* context? (KL-max)**

- [ ] **Exp 6**: Exp 3 + KL-max reward
  - Pretrained HL
  - LL trained with GRPO
  - Reward = task + λ·(log p(y|x,c) − log q(y|x))
  - **Purpose**: detect context reliance beyond prompt length

- [ ] **Exp 7**: Exp 5 + KL-max reward
  - SFT HL
  - LL trained with GRPO + KL-max
  - **Purpose**: main "context utilization" result

- [ ] **Exp 8**: Exp 7, λ = 0
  - Same setup, KL term removed
  - **Purpose**: clean ablation → gains should vanish

- [ ] **Exp 9**: Exp 7, different prior q(y|x)
  - q = frozen initial LL
  - q = empty-context LL snapshot
  - **Purpose**: robustness of utilization signal

---

### **PHASE 3 — Stability & training structure**

- [ ] **Exp 10**: Exp 7, no GRPO KL-to-reference
  - β_LL = 0
  - **Purpose**: over-optimization / collapse check

- [ ] **Exp 11**: Frozen LL, train HL only
  - HL (SFT or RLHF)
  - LL never updated
  - **Purpose**: separate "learn demos" vs "learn to use demos"

- [ ] **Exp 12**: Alternating HL–LL training
  - LL: GRPO (+ KL-max)
  - HL: RLHF (confidence + task)
  - Multiple alternations
  - **Purpose**: full method

- [ ] **Exp 13**: Joint HL–LL training (no alternation)
  - **Purpose**: stability ablation

---

### **PHASE 4 — Sanity & stress tests**

- [ ] **Exp 14**: Random / mismatched contexts
  - Same prompt length as Exp 7
  - **Purpose**: verify signal is semantic, not length

- [ ] **Exp 15**: Train with context, eval without context
  - **Purpose**: dependency & over-reliance check

- [ ] **Exp 16**: Multiple contexts (k > 1)
  - **Purpose**: scalability of utilization objective

---

## Logging Schema

### 1. Run-level logs (one JSON per run)

**Path**: `logs/exp05_sft_train_ctx/run_meta.json`

```json
{
  "exp_id": "exp05",
  "hl_type": "sft",
  "ll_algo": "grpo",
  "kl_max": false,
  "lambda": 0.0,
  "prior": "none",
  "dataset": "math500",
  "group_size": 8,
  "seed": 42
}
```

### 2. Per-sample logs (JSONL, most important)

**Path**: `logs/exp07_klmax/samples.jsonl`

```json
{
  "uid": "math_237",
  "x_id": 237,
  "has_context": true,
  "context_id": "ctx_237_a",
  "reward_task": 1.0,
  "reward_total": 1.43,
  "advantage": 0.62,
  "logp_ctx": -12.4,
  "logp_noctx": -15.9,
  "delta_logp": 3.5,
  "answer_correct": true,
  "response_len": 312
}
```

### 3. Aggregate metrics (auto-derived)

**Path**: `logs/exp07_klmax/metrics.json`

```json
{
  "accuracy": 0.412,
  "mean_delta_logp": 2.87,
  "pct_positive_delta_logp": 0.78,
  "mean_advantage": 0.04,
  "ece": 0.09
}
```

---

## Directory Structure

```
logs/
  exp01_grpo_base/
  exp02_pre_test_ctx/
  exp03_pre_train_ctx/
  exp04_sft_test_ctx/
  exp05_sft_train_ctx/
  exp06_pre_klmax/
  exp07_sft_klmax/
  exp08_sft_klmax_l0/
  exp09_sft_klmax_altprior/
  exp12_full_alt/
```

---

## Key Success Signals

If the method is **working**, you should see:

- [ ] Exp 5 > Exp 3 > Exp 1 (accuracy)
- [ ] Exp 7 > Exp 5 (accuracy + stability)
- [ ] **Δlogp(y|x,c) − logp(y|x)** positive in Exp 7
- [ ] Random contexts (Exp 14) collapse accuracy
- [ ] Removing KL-max (Exp 8) collapses Δlogp signal

---

## Phase Dependencies

- **Phase 0** → **Phase 1** → **Phase 2** → **Phase 3** → **Phase 4**
- Each phase builds on previous results
- Start with Phase 0 (Exp 1) for baseline
- Phase 2 experiments are the core contribution
- Phase 3-4 are robustness checks

---

## Implementation Notes

- All experiments use existing `verl` GRPO infrastructure
- Minimal extra machinery needed beyond reward shaping
- KL-max term implemented as reward addition in GRPO
- Logging can be added to existing rollout/callback hooks
- Context generation handled by HL model (pretrained or SFT)
