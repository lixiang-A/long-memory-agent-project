from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer

from alignment_course.io_utils import (
    count_trainable_parameters,
    read_jsonl,
    save_json,
    set_seed,
    timed,
)
from alignment_course.model_utils import default_torch_dtype, load_tokenizer, make_lora_config
from alignment_course.training_utils import build_config, trainer_tokenizer_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward model training on chosen/rejected pairs.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.base_model)
    def to_reward_rows(rows):
        return [
            {
                "source_idx": row["source_idx"],
                "chosen": row["prompt"] + row["chosen"],
                "rejected": row["prompt"] + row["rejected"],
            }
            for row in rows
        ]

    train_ds = Dataset.from_list(to_reward_rows(read_jsonl(args.train_path, args.max_train_samples)))
    eval_ds = Dataset.from_list(to_reward_rows(read_jsonl(args.eval_path, args.max_eval_samples)))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        torch_dtype=default_torch_dtype(),
        trust_remote_code=True,
    )
    model.config.use_cache = False

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
        "max_length": args.max_seq_length,
    }
    reward_config = build_config(RewardConfig, config_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": reward_config,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "peft_config": make_lora_config("SEQ_CLS"),
    }
    trainer_kwargs.update(trainer_tokenizer_kwargs(RewardTrainer, tokenizer))
    trainer = RewardTrainer(**trainer_kwargs)

    with timed() as timing:
        result = trainer.train()

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    save_json(
        args.output_dir / "run_summary.json",
        {
            "method": "reward_model_lora",
            "base_model": args.base_model,
            "train_rows": len(train_ds),
            "eval_rows": len(eval_ds),
            "max_steps": args.max_steps,
            "train_runtime_seconds": timing["seconds"],
            "train_result": result.metrics,
            "parameters": count_trainable_parameters(trainer.model),
            "final_dir": str(final_dir),
        },
    )
    print(f"Reward model adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
