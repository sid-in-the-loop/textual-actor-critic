import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


class HFContextGenerator:
    """Lightweight wrapper around a HuggingFace causal LM used as the higher-level model."""

    def __init__(self, cfg: dict[str, Any]):
        self.enabled = bool(cfg and cfg.get("path"))
        if not self.enabled:
            self.model = None
            self.tokenizer = None
            return

        model_path = cfg["path"]
        tokenizer_path = cfg.get("tokenizer_path", model_path)
        device = cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.max_new_tokens = 2048
        self.temperature = cfg.get("temperature", 0.7)
        self.top_p = cfg.get("top_p", 0.95)
        self.top_k = cfg.get("top_k", 0)
        self.do_sample = cfg.get("do_sample", True)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=cfg.get("trust_remote_code", False))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=cfg.get("trust_remote_code", False),
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompts: Sequence[str], num_samples: int) -> list[list[str]]:
        if not self.enabled or self.model is None:
            return [[] for _ in prompts]

        all_contexts: list[list[str]] = []
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.top_k > 0:
            generation_kwargs["top_k"] = self.top_k

        with torch.inference_mode():
            for prompt in prompts:
                contexts_for_prompt = []
                for _ in range(num_samples):
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    output = self.model.generate(**inputs, **generation_kwargs)
                    generated_part = output[:, inputs["input_ids"].shape[-1] :]
                    text = self.tokenizer.batch_decode(generated_part, skip_special_tokens=True)[0].strip()
                    contexts_for_prompt.append(text)
                all_contexts.append(contexts_for_prompt)
        return all_contexts


class HierarchicalContextManager:
    """Handles prompt augmentation with contexts and enforces token limits."""

    def __init__(self, cfg: DictConfig | None, tokenizer):
        cfg = cfg or DictConfig({})
        self.enabled = bool(cfg.get("enabled", False))
        self.context_source = cfg.get("context_source", "none")
        self.num_context_samples = int(cfg.get("num_context_samples", 1))
        self.max_prompt_tokens = int(cfg.get("max_prompt_tokens", 2048))
        self.context_header = cfg.get("context_header", "### Retrieved demonstrations")
        self.context_prefix = cfg.get("context_prefix", "Example")
        self.prompt_template = cfg.get(
            "prompt_template",
            "Given the following math problem:\n{question}\n\nGenerate a related worked example with solution.",
        )
        self.ll_trainable = bool(cfg.get("ll_trainable", True))
        self.tokenizer = tokenizer
        self.cache: dict[str, list[str]] = {}

        hl_model_cfg = cfg.get("hl_model", {})
        self.generator = HFContextGenerator(hl_model_cfg)

    def prepare_batch(self, batch: DataProto, split: str) -> DataProto:
        if not self.enabled:
            if self.max_prompt_tokens:
                self._truncate_existing(batch)
            return batch

        prompts = self._extract_prompts(batch)
        sample_ids = self._extract_sample_ids(batch)

        contexts: list[list[str]] = [[] for _ in prompts]
        if self.context_source != "none":
            contexts = self._ensure_contexts(prompts, sample_ids)

        augmented_prompts = self._compose_prompts(prompts, contexts)
        encoded = self.tokenizer(
            augmented_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens,
            return_tensors="pt",
            add_special_tokens=False,
        )

        batch.batch["input_ids"] = encoded["input_ids"]
        batch.batch["attention_mask"] = encoded["attention_mask"]
        batch.batch["position_ids"] = compute_position_id_with_mask(encoded["attention_mask"])
        batch.batch["raw_prompt_ids"] = np.array([ids.tolist() for ids in encoded["input_ids"]], dtype=object)

        batch.non_tensor_batch["full_prompts"] = np.array(augmented_prompts, dtype=object)
        batch.non_tensor_batch["hierarchical/base_prompts"] = np.array(prompts, dtype=object)
        batch.non_tensor_batch["hierarchical/contexts"] = np.array(contexts, dtype=object)
        batch.non_tensor_batch["hierarchical/sample_ids"] = np.array(sample_ids, dtype=object)
        batch.non_tensor_batch["hierarchical/split"] = np.array([split] * len(sample_ids), dtype=object)

        return batch

    def _extract_prompts(self, batch: DataProto) -> list[str]:
        if "full_prompts" in batch.non_tensor_batch:
            return batch.non_tensor_batch["full_prompts"].tolist()
        return self.tokenizer.batch_decode(batch.batch["input_ids"], skip_special_tokens=False)

    def _extract_sample_ids(self, batch: DataProto) -> list[str]:
        data_sources = batch.non_tensor_batch.get("data_source")
        indices = batch.non_tensor_batch.get("index")
        sample_ids = []
        for i in range(len(batch.batch["input_ids"])):
            src = data_sources[i] if data_sources is not None else "unknown"
            idx = indices[i] if indices is not None else i
            sample_ids.append(f"{src}:{idx}")
        return sample_ids

    def _ensure_contexts(self, prompts: list[str], sample_ids: list[str]) -> list[list[str]]:
        missing_prompts = []
        missing_ids = []
        contexts: list[list[str]] = []

        for prompt, sid in zip(prompts, sample_ids, strict=True):
            if sid in self.cache:
                contexts.append(self.cache[sid])
            else:
                missing_prompts.append(prompt)
                missing_ids.append(sid)
                contexts.append([])

        if missing_prompts and self.generator.enabled:
            generated = self.generator.generate(
                [self.prompt_template.format(question=q) for q in missing_prompts],
                self.num_context_samples,
            )
            for sid, ctx in zip(missing_ids, generated, strict=True):
                self.cache[sid] = ctx

            gen_iter = iter(generated)
            for idx, ctx in enumerate(contexts):
                if not ctx:
                    contexts[idx] = next(gen_iter)

        return contexts

    def _compose_prompts(self, prompts: list[str], contexts: list[list[str]]) -> list[str]:
        augmented = []
        for base, ctx_list in zip(prompts, contexts, strict=True):
            if not ctx_list:
                augmented.append(base)
                continue
            context_blocks = "\n\n".join(
                f"{self.context_prefix} {i+1}:\n{ctx.strip()}" for i, ctx in enumerate(ctx_list)
            )
            augmented_prompt = f"{self.context_header}\n{context_blocks}\n\n{base}"
            augmented.append(augmented_prompt)
        return augmented

    def _truncate_existing(self, batch: DataProto):
        input_ids = batch.batch["input_ids"]
        if input_ids.shape[-1] <= self.max_prompt_tokens:
            return
        keep = self.max_prompt_tokens
        batch.batch["input_ids"] = input_ids[:, -keep:]
        if "attention_mask" in batch.batch:
            batch.batch["attention_mask"] = batch.batch["attention_mask"][:, -keep:]
        else:
            attention_mask = torch.ones_like(batch.batch["input_ids"])
            batch.batch["attention_mask"] = attention_mask
        batch.batch["position_ids"] = compute_position_id_with_mask(batch.batch["attention_mask"])
        batch.batch["raw_prompt_ids"] = np.array(
            [ids.tolist() for ids in batch.batch["input_ids"]], dtype=object
        )


class HierarchicalLogger:
    """Writes per-question JSONL logs with context utilization statistics."""

    def __init__(self, cfg: DictConfig | None, model_name: str, experiment_name: str, tokenizer, repeat_n: int):
        cfg = cfg or DictConfig({})
        self.enabled = bool(cfg.get("logging_enabled", cfg.get("enabled", False)))
        self.experiment_id = cfg.get("experiment_id", "default")
        self.base_dir = Path(cfg.get("base_dir", "logs")) / self.experiment_id
        self.model_name = cfg.get("model_name", model_name)
        self.tokenizer = tokenizer
        self.repeat_n = max(1, repeat_n)
        
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def log_run_meta(self, config):
        if not self.enabled:
            return
        
        from omegaconf import OmegaConf
        import json
        
        meta_path = self.base_dir / "run_meta.json"
        meta = {
            "exp_id": self.experiment_id,
            "hl_type": config.hierarchical.get("context_source", "none"),
            "ll_algo": config.algorithm.adv_estimator,
            "kl_max": config.hierarchical.kl_max.get("enabled", False),
            "lambda": config.hierarchical.kl_max.get("lambda_coef", 0.0),
            "prior": config.hierarchical.kl_max.get("reference_prior", "none"),
            "dataset": config.env.get("dataset", "math500"),
            "group_size": config.env.rollout.n,
            "seed": config.env.seed,
            "config": OmegaConf.to_container(config, resolve=True)
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def log_metrics(self, metrics: dict):
        if not self.enabled:
            return
        
        import json
        metrics_path = self.base_dir / "metrics.json"
        
        # Extract requested metrics if they exist
        payload = {
            "accuracy": metrics.get("val/success_rate", 0.0),
            "mean_reward": metrics.get("val/reward", 0.0),
            "mean_response_len": metrics.get("val/response_length/mean", 0.0),
            "raw": metrics
        }
        
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

    def log_batch(self, split: str, batch: DataProto, step: int):
        if not self.enabled:
            return
        if "responses" not in batch.batch:
            return

        import numpy as np
        import json
        import os

        contexts = batch.non_tensor_batch.get("hierarchical/contexts")
        base_prompts = batch.non_tensor_batch.get("hierarchical/base_prompts")
        sample_ids = batch.non_tensor_batch.get("hierarchical/sample_ids")

        prompts = batch.batch.get("prompts", None)
        if prompts is None:
            prompts = batch.batch["input_ids"]
        prompt_texts = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)
        response_texts = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

        token_level_scores = batch.batch.get("token_level_scores", None)
        token_level_rewards = batch.batch.get("token_level_rewards", token_level_scores)
        advantages = batch.batch.get("advantages", None)
        response_mask = batch.batch.get("response_mask", None)
        old_log_probs = batch.batch.get("old_log_probs", None)
        ref_log_prob = batch.batch.get("ref_log_prob", None)

        uids = batch.non_tensor_batch.get("uid")
        counters: dict[Any, int] = {}

        samples_path = self.base_dir / f"step_{step}.jsonl"

        for idx, (prompt_text, response_text) in enumerate(zip(prompt_texts, response_texts, strict=True)):
            uid = uids[idx] if uids is not None else idx
            repeat_idx = counters.get(uid, 0)
            counters[uid] = (repeat_idx + 1) % self.repeat_n

            reward_task = (
                token_level_scores[idx].sum().item() if token_level_scores is not None else 0.0
            )
            reward_total = (
                token_level_rewards[idx].sum().item() if token_level_rewards is not None else reward_task
            )

            adv_value = None
            if advantages is not None and response_mask is not None:
                adv_value = (
                    (advantages[idx] * response_mask[idx]).sum().item()
                    / (response_mask[idx].sum().item() + 1e-6)
                )

            import math
            logp_ctx = None
            if old_log_probs is not None and response_mask is not None:
                val = (old_log_probs[idx] * response_mask[idx]).sum().item()
                logp_ctx = val if not math.isnan(val) else None

            logp_noctx = None
            if ref_log_prob is not None and response_mask is not None:
                val = (ref_log_prob[idx] * response_mask[idx]).sum().item()
                logp_noctx = val if not math.isnan(val) else None

            delta_logp = logp_ctx - logp_noctx if logp_ctx is not None and logp_noctx is not None else None

            response_len = int(response_mask[idx].sum().item()) if response_mask is not None else len(
                batch.batch["responses"][idx]
            )

            context_entry = contexts[idx] if contexts is not None else []
            if isinstance(context_entry, np.ndarray):
                context_values = context_entry.tolist()
            elif isinstance(context_entry, (list, tuple)):
                context_values = list(context_entry)
            else:
                context_values = [context_entry] if context_entry else []

            base_prompt_entry = base_prompts[idx] if base_prompts is not None else prompt_text

            payload = {
                "global_step": step,
                "split": split,
                "uid": str(uid),
                "k_index": repeat_idx,
                "question": base_prompt_entry,
                "context": context_values,
                "answer": response_text,
                "reward_task": reward_task,
                "reward_total": reward_total,
                "advantage": adv_value,
                "logp_ctx": logp_ctx,
                "logp_noctx": logp_noctx,
                "delta_logp": delta_logp,
                "response_len": response_len,
                "answer_correct": True if reward_task > 0 else False,
            }

            with open(samples_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + os.linesep)


