import os
from typing import List, Dict, Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer


class RobertaValueWorker:
    def __init__(self, config):
        self.config = config
        use_gpu = config.get("use_gpu", False)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.value_head = None
        self.tokenizer = None
        self.optimizer = None
        self.max_length = config.get("max_length", 512)

    def init_model(self):
        model_path = self.config.get("model_path", "roberta-base")
        tokenizer_path = self.config.get("tokenizer_path") or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        base_model = AutoModel.from_pretrained(model_path)
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
        self.model = base_model.to(self.device)
        self.value_head = self.value_head.to(self.device)

        if not self.config.get("freeze", False):
            lr = self.config.get("lr", 3e-5)
            weight_decay = self.config.get("weight_decay", 0.01)
            params = list(self.model.parameters()) + list(self.value_head.parameters())
            self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.value_head.parameters():
                param.requires_grad = False
            self.optimizer = None

    def _encode(self, texts: List[str]):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    @torch.no_grad()
    def compute_values(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, device=self.device)
        inputs = self._encode(texts)
        outputs = self.model(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]
        values = self.value_head(pooled).squeeze(-1)
        return values

    def update_value_fn(self, texts: List[str], targets: List[float]) -> Dict[str, float]:
        if self.optimizer is None or not texts:
            return {"value_fn/loss": 0.0}
        inputs = self._encode(texts)
        outputs = self.model(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]
        preds = self.value_head(pooled).squeeze(-1)
        target_tensor = torch.tensor(targets, device=self.device)
        loss = nn.functional.mse_loss(preds, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"value_fn/loss": loss.item()}

    def save_checkpoint(self, local_path: str, remote_path: str = None, global_steps: int = 0, max_ckpt_to_keep: int = None):
        os.makedirs(local_path, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "steps": global_steps,
        }, os.path.join(local_path, "value_fn.pt"))

    def load_checkpoint(self, path: str, del_local_after_load: bool = False):
        ckpt_file = os.path.join(path, "value_fn.pt")
        if not os.path.exists(ckpt_file):
            return
        state = torch.load(ckpt_file, map_location=self.device)
        self.init_model()
        self.model.load_state_dict(state["model"])
        self.value_head.load_state_dict(state["value_head"])
        if self.optimizer is not None and state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
