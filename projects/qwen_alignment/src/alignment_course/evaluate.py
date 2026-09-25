from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from alignment_course.io_utils import get_device_name, read_jsonl, save_json, set_seed, timed
from alignment_course.model_utils import load_policy_model, load_reward_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DPO vs PPO/RLHF pilot.")
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--dpo-model", required=True)
    parser.add_argument("--dpo-base", required=True)
    parser.add_argument("--ppo-model")
    parser.add_argument("--ppo-base")
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--reward-base", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=50)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _move(model: Any, device: str) -> Any:
    model.to(device)
    model.eval()
    return model


def _encode_pair(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_seq_length: int,
    max_prompt_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids[-max_prompt_length:]
    response_ids = tokenizer(response + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    available = max_seq_length - len(prompt_ids)
    response_ids = response_ids[: max(1, available)]
    input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return input_ids, attention_mask, labels


@torch.no_grad()
def sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: str,
    max_seq_length: int,
    max_prompt_length: int,
) -> float:
    input_ids, attention_mask, labels = _encode_pair(
        tokenizer, prompt, response, max_seq_length, max_prompt_length
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs[mask].sum().detach().cpu())


@torch.no_grad()
def approximate_kl_to_reference(
    model: Any,
    reference: Any,
    tokenizer: Any,
    rows: List[Dict[str, Any]],
    device: str,
    max_seq_length: int,
    max_prompt_length: int,
) -> float:
    values = []
    for row in tqdm(rows, desc="approx_kl", leave=False):
        input_ids, attention_mask, labels = _encode_pair(
            tokenizer,
            row["prompt"],
            row["chosen"],
            max_seq_length,
            max_prompt_length,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        ref_logits = reference(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        mask = labels[:, 1:] != -100
        logp = F.log_softmax(logits.float(), dim=-1)
        ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
        p = logp.exp()
        token_kl = (p * (logp - ref_logp)).sum(dim=-1)
        if mask.any():
            values.append(float(token_kl[mask].mean().detach().cpu()))
    return float(np.mean(values)) if values else float("nan")


def preference_accuracy(
    model: Any,
    tokenizer: Any,
    rows: List[Dict[str, Any]],
    device: str,
    max_seq_length: int,
    max_prompt_length: int,
) -> Dict[str, float]:
    correct = 0
    margins = []
    for row in tqdm(rows, desc="preference_accuracy", leave=False):
        chosen_lp = sequence_logprob(
            model, tokenizer, row["prompt"], row["chosen"], device, max_seq_length, max_prompt_length
        )
        rejected_lp = sequence_logprob(
            model, tokenizer, row["prompt"], row["rejected"], device, max_seq_length, max_prompt_length
        )
        margin = chosen_lp - rejected_lp
        correct += int(margin > 0)
        margins.append(margin)
    return {
        "accuracy": float(correct / len(rows)) if rows else float("nan"),
        "mean_logprob_margin": float(np.mean(margins)) if margins else float("nan"),
        "std_logprob_margin": float(np.std(margins)) if margins else float("nan"),
    }


@torch.no_grad()
def reward_score(
    reward_model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: str,
    max_seq_length: int,
) -> float:
    encoded = tokenizer(
        prompt + response,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = reward_model(**encoded)
    logits = outputs.logits
    return float(logits.reshape(-1)[-1].detach().cpu())


@torch.no_grad()
def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_prompt_length: int,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def win_rate_against_reward_model(
    dpo_model: Any,
    ppo_model: Optional[Any],
    reward_model: Any,
    tokenizer: Any,
    rows: List[Dict[str, Any]],
    device: str,
    max_seq_length: int,
    max_prompt_length: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    if ppo_model is None:
        return {"status": "skipped_no_ppo_model"}
    dpo_wins = 0
    ppo_wins = 0
    ties = 0
    dpo_scores = []
    ppo_scores = []
    for row in tqdm(rows, desc="rm_win_rate", leave=False):
        dpo_response = generate_response(
            dpo_model, tokenizer, row["prompt"], device, max_prompt_length, max_new_tokens
        )
        ppo_response = generate_response(
            ppo_model, tokenizer, row["prompt"], device, max_prompt_length, max_new_tokens
        )
        dpo_score = reward_score(
            reward_model, tokenizer, row["prompt"], dpo_response, device, max_seq_length
        )
        ppo_score = reward_score(
            reward_model, tokenizer, row["prompt"], ppo_response, device, max_seq_length
        )
        dpo_scores.append(dpo_score)
        ppo_scores.append(ppo_score)
        if dpo_score > ppo_score:
            dpo_wins += 1
        elif ppo_score > dpo_score:
            ppo_wins += 1
        else:
            ties += 1
    n = len(rows)
    return {
        "status": "completed",
        "n": n,
        "dpo_win_rate": float(dpo_wins / n) if n else float("nan"),
        "ppo_win_rate": float(ppo_wins / n) if n else float("nan"),
        "tie_rate": float(ties / n) if n else float("nan"),
        "dpo_mean_reward": float(np.mean(dpo_scores)) if dpo_scores else float("nan"),
        "ppo_mean_reward": float(np.mean(ppo_scores)) if ppo_scores else float("nan"),
    }


def maybe_load_ppo(args: argparse.Namespace, device: str) -> Optional[Any]:
    if not args.ppo_model:
        return None
    model_path = Path(args.ppo_model)
    if not model_path.exists():
        return None
    if (model_path / "adapter_config.json").exists() and not args.ppo_base:
        return None
    return _move(load_policy_model(args.ppo_model, base_model=args.ppo_base), device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rows = read_jsonl(args.test_path, args.max_pairs)
    generation_rows = rows[: args.num_generations]
    device = get_device_name()

    tokenizer = load_tokenizer(args.reference_model)
    with timed() as timing:
        reference = _move(load_policy_model(args.reference_model), device)
        dpo_model = _move(load_policy_model(args.dpo_model, base_model=args.dpo_base), device)
        ppo_model = maybe_load_ppo(args, device)
        reward_tokenizer = load_tokenizer(args.reward_model, fallback_model=args.reward_base)
        reward_model = _move(
            load_reward_model(args.reward_model, base_model=args.reward_base),
            device,
        )

        metrics = {
            "device": device,
            "n_preference_pairs": len(rows),
            "n_generation_prompts": len(generation_rows),
            "reference_preference": preference_accuracy(
                reference,
                tokenizer,
                rows,
                device,
                args.max_seq_length,
                args.max_prompt_length,
            ),
            "dpo_preference": preference_accuracy(
                dpo_model,
                tokenizer,
                rows,
                device,
                args.max_seq_length,
                args.max_prompt_length,
            ),
            "dpo_approx_kl_to_reference": approximate_kl_to_reference(
                dpo_model,
                reference,
                tokenizer,
                rows,
                device,
                args.max_seq_length,
                args.max_prompt_length,
            ),
            "rm_win_rate": win_rate_against_reward_model(
                dpo_model,
                ppo_model,
                reward_model,
                reward_tokenizer,
                generation_rows,
                device,
                args.max_seq_length,
                args.max_prompt_length,
                args.max_new_tokens,
            ),
        }
        if ppo_model is not None:
            metrics["ppo_preference"] = preference_accuracy(
                ppo_model,
                tokenizer,
                rows,
                device,
                args.max_seq_length,
                args.max_prompt_length,
            )
            metrics["ppo_approx_kl_to_reference"] = approximate_kl_to_reference(
                ppo_model,
                reference,
                tokenizer,
                rows,
                device,
                args.max_seq_length,
                args.max_prompt_length,
            )

    metrics["runtime_seconds"] = timing["seconds"]
    save_json(args.output_json, metrics)
    print(f"Evaluation metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
