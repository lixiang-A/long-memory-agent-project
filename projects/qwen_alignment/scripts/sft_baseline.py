#!/usr/bin/env python3
"""Quick SFT baseline: compute preference accuracy of SFT model on test set.
Pure inference, no training. Should finish in <10 minutes on M3."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

import torch
from alignment_course.io_utils import get_device_name, read_jsonl, save_json, set_seed, timed
from alignment_course.model_utils import load_policy_model, load_tokenizer

# Reuse the same preference_accuracy function
from alignment_course.evaluate import preference_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=Path, default=PROJECT / "data/processed/hh_test.jsonl")
    parser.add_argument("--sft-model", type=str, default=str(PROJECT / "artifacts/main/sft_merged"))
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--output-json", type=Path, default=PROJECT / "artifacts/main/sft_baseline_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = read_jsonl(args.test_path, args.max_pairs)
    device = get_device_name()

    print(f"Device: {device}")
    print(f"Test pairs: {len(rows)}")
    print(f"Loading SFT model from: {args.sft_model}")

    with timed() as timing:
        tokenizer = load_tokenizer(args.sft_model)
        sft_model = load_policy_model(args.sft_model)
        sft_model.to(device)
        sft_model.eval()

        sft_pref = preference_accuracy(
            sft_model, tokenizer, rows, device,
            args.max_seq_length, args.max_prompt_length,
        )

    result = {
        "model": str(args.sft_model),
        "n_pairs": len(rows),
        "device": device,
        "sft_preference_accuracy": sft_pref,
        "runtime_seconds": timing["seconds"],
    }
    save_json(args.output_json, result)
    print(f"\nSFT preference accuracy: {sft_pref['accuracy']:.4f}")
    print(f"SFT mean logprob margin: {sft_pref['mean_logprob_margin']:.2f}")
    print(f"SFT std logprob margin:  {sft_pref['std_logprob_margin']:.2f}")
    print(f"Runtime: {timing['seconds']:.0f}s")
    print(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    main()
