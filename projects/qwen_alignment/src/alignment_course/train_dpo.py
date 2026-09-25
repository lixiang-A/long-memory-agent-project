from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from peft import get_peft_model
from trl import DPOConfig
import trl.import_utils as trl_import_utils

trl_import_utils._llm_blender_available = False
trl_import_utils._weave_available = False
from trl import DPOTrainer

from alignment_course.io_utils import (
    count_trainable_parameters,
    read_jsonl,
    save_json,
    set_seed,
    timed,
)
from alignment_course.model_utils import load_policy_model, load_tokenizer, make_lora_config
from alignment_course.training_utils import build_config, trainer_tokenizer_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training with LoRA.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_name_or_path)
    train_ds = Dataset.from_list(read_jsonl(args.train_path, args.max_train_samples))
    eval_ds = Dataset.from_list(read_jsonl(args.eval_path, args.max_eval_samples))
    model = load_policy_model(args.model_name_or_path)
    model.config.use_cache = False
    model = get_peft_model(model, make_lora_config("CAUSAL_LM"))
    model.warnings_issued = {}

    config_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": 5,
        "save_steps": max(args.max_steps, 1),
        "eval_strategy": "steps",
        "eval_steps": max(args.max_steps, 1),
        "report_to": [],
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "bf16": False,
        "fp16": False,
        "beta": args.beta,
        "max_length": args.max_seq_length,
        "max_prompt_length": args.max_prompt_length,
    }
    dpo_config = build_config(DPOConfig, config_kwargs)

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_config,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    trainer_kwargs.update(trainer_tokenizer_kwargs(DPOTrainer, tokenizer))
    trainer = DPOTrainer(**trainer_kwargs)

    with timed() as timing:
        result = trainer.train()

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    save_json(
        args.output_dir / "run_summary.json",
        {
            "method": "dpo_lora",
            "model_name_or_path": args.model_name_or_path,
            "train_rows": len(train_ds),
            "eval_rows": len(eval_ds),
            "max_steps": args.max_steps,
            "beta": args.beta,
            "train_runtime_seconds": timing["seconds"],
            "train_result": result.metrics,
            "parameters": count_trainable_parameters(trainer.model),
            "final_dir": str(final_dir),
        },
    )
    print(f"DPO adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
