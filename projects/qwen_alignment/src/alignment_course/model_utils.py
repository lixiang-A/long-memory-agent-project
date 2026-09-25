from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def is_peft_adapter(path: Optional[str]) -> bool:
    if not path:
        return False
    return (Path(path) / "adapter_config.json").exists()


def default_torch_dtype() -> Any:
    if torch.cuda.is_available():
        return torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer(model_or_path: str, fallback_model: Optional[str] = None) -> Any:
    path = Path(model_or_path)
    source = model_or_path
    if path.exists() and is_peft_adapter(model_or_path):
        if fallback_model is None:
            raise ValueError("Tokenizer fallback_model is required for PEFT adapter paths.")
        source = fallback_model
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def make_lora_config(task_type: str = "CAUSAL_LM") -> LoraConfig:
    kwargs = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": task_type,
        "target_modules": DEFAULT_LORA_TARGETS,
    }
    if task_type == "SEQ_CLS":
        kwargs["modules_to_save"] = ["score"]
    return LoraConfig(**kwargs)


def load_policy_model(
    model_or_adapter: str,
    base_model: Optional[str] = None,
    trainable_adapter: bool = False,
) -> Any:
    dtype = default_torch_dtype()
    if is_peft_adapter(model_or_adapter):
        if base_model is None:
            raise ValueError("base_model is required when loading a PEFT policy adapter.")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return PeftModel.from_pretrained(base, model_or_adapter, is_trainable=trainable_adapter)
    return AutoModelForCausalLM.from_pretrained(
        model_or_adapter,
        torch_dtype=dtype,
        trust_remote_code=True,
    )


def load_reward_model(
    model_or_adapter: str,
    base_model: Optional[str] = None,
    trainable_adapter: bool = False,
) -> Any:
    dtype = default_torch_dtype()
    if is_peft_adapter(model_or_adapter):
        if base_model is None:
            raise ValueError("base_model is required when loading a PEFT reward adapter.")
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=1,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return PeftModel.from_pretrained(base, model_or_adapter, is_trainable=trainable_adapter)
    return AutoModelForSequenceClassification.from_pretrained(
        model_or_adapter,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

