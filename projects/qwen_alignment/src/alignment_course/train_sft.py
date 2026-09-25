from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from alignment_course.io_utils import (
    count_trainable_parameters,
    read_jsonl,
    save_json,
    set_seed,
    timed,
)
from alignment_course.model_utils import default_torch_dtype, load_tokenizer, make_lora_config
from alignment_course.training_utils import build_config, supports_parameter, trainer_tokenizer_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT warm start on chosen HH-RLHF responses.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
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
    train_rows = read_jsonl(args.train_path, args.max_train_samples)
    eval_rows = read_jsonl(args.eval_path, args.max_eval_samples)

    eos = tokenizer.eos_token or ""

    def to_sft_rows(rows):
        tokenized_rows = []
        for row in rows:
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids
            completion_ids = tokenizer(row["chosen"] + eos, add_special_tokens=False).input_ids
            if len(completion_ids) >= args.max_seq_length:
                input_ids = completion_ids[: args.max_seq_length]
                completion_mask = [1] * len(input_ids)
            else:
                prompt_budget = args.max_seq_length - len(completion_ids)
                prompt_ids = prompt_ids[-prompt_budget:]
                input_ids = prompt_ids + completion_ids
                completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            tokenized_rows.append(
                {
                    "source_idx": row["source_idx"],
                    "input_ids": input_ids,
                    "completion_mask": completion_mask,
                }
            )
        return tokenized_rows

    train_ds = Dataset.from_list(to_sft_rows(train_rows))
    eval_ds = Dataset.from_list(to_sft_rows(eval_rows))

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=default_torch_dtype(),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(model, make_lora_config("CAUSAL_LM"))

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
        "max_length": args.max_seq_length,
        "completion_only_loss": True,
        "bf16": False,
        "fp16": False,
    }
    sft_config = build_config(SFTConfig, config_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    trainer_kwargs.update(trainer_tokenizer_kwargs(SFTTrainer, tokenizer))
    if not supports_parameter(SFTConfig, "max_length") and supports_parameter(SFTTrainer, "max_seq_length"):
        trainer_kwargs["max_seq_length"] = args.max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)
    with timed() as timing:
        result = trainer.train()

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    save_json(
        args.output_dir / "run_summary.json",
        {
            "method": "sft_lora",
            "base_model": args.base_model,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "max_steps": args.max_steps,
            "train_runtime_seconds": timing["seconds"],
            "train_result": result.metrics,
            "parameters": count_trainable_parameters(model),
            "final_dir": str(final_dir),
        },
    )
    print(f"SFT adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
