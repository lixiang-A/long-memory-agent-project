from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForSequenceClassification
from trl import PPOConfig, PPOTrainer

from alignment_course.io_utils import (
    count_trainable_parameters,
    read_jsonl,
    save_json,
    set_seed,
    timed,
)
from alignment_course.model_utils import (
    default_torch_dtype,
    load_policy_model,
    load_reward_model,
    load_tokenizer,
    make_lora_config,
)
from alignment_course.training_utils import build_config, trainer_tokenizer_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short PPO-RLHF pilot. Designed to be optional.")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--reward-base", required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--total-episodes", type=int, default=128)
    parser.add_argument("--response-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = load_tokenizer(args.policy_model)
        rows = read_jsonl(args.train_path, args.max_prompts)
        prompt_ds = Dataset.from_list(
            [
                {
                    "input_ids": tokenizer(row["prompt"], truncation=True, max_length=512).input_ids,
                }
                for row in rows
            ]
        )

        model = load_policy_model(args.policy_model)
        model.config.use_cache = False
        model = get_peft_model(model, make_lora_config("CAUSAL_LM"))
        ref_model = load_policy_model(args.policy_model)
        reward_model = load_reward_model(args.reward_model, base_model=args.reward_base)
        value_model = AutoModelForSequenceClassification.from_pretrained(
            args.policy_model,
            num_labels=1,
            torch_dtype=default_torch_dtype(),
            trust_remote_code=True,
        )

        config_kwargs = {
            "output_dir": str(args.output_dir),
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": 1,
            "total_episodes": args.total_episodes,
            "response_length": args.response_length,
            "logging_steps": 5,
            "report_to": [],
        }
        ppo_config = build_config(PPOConfig, config_kwargs)
        trainer_kwargs = {
            "args": ppo_config,
            "model": model,
            "ref_model": ref_model,
            "reward_model": reward_model,
            "value_model": value_model,
            "train_dataset": prompt_ds,
            "eval_dataset": prompt_ds,
        }
        trainer_kwargs.update(trainer_tokenizer_kwargs(PPOTrainer, tokenizer))

        trainer = PPOTrainer(**trainer_kwargs)
        with timed() as timing:
            train_result = trainer.train()

        final_dir = args.output_dir / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        save_json(
            args.output_dir / "ppo_pilot_status.json",
            {
                "status": "completed",
                "method": "ppo_pilot_lora",
                "policy_model": args.policy_model,
                "reward_model": args.reward_model,
                "prompt_rows": len(prompt_ds),
                "total_episodes": args.total_episodes,
                "response_length": args.response_length,
                "train_runtime_seconds": timing["seconds"],
                "train_result": getattr(train_result, "metrics", {}),
                "parameters": count_trainable_parameters(trainer.model),
                "final_dir": str(final_dir),
            },
        )
        print(f"PPO pilot saved to {final_dir}")
    except Exception as exc:
        save_json(
            args.output_dir / "ppo_pilot_status.json",
            {
                "status": "failed",
                "reason": repr(exc),
                "interpretation": (
                    "This is an allowed outcome for the lightweight RLHF pilot. "
                    "Report it as a compute/API/stability limitation rather than a completed PPO run."
                ),
            },
        )
        raise


if __name__ == "__main__":
    main()
