from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM

from alignment_course.io_utils import save_json
from alignment_course.model_utils import default_torch_dtype, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA policy adapter into a full model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=default_torch_dtype(),
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(args.output_dir), safe_serialization=True)
    tokenizer = load_tokenizer(args.base_model)
    tokenizer.save_pretrained(str(args.output_dir))
    save_json(
        args.output_dir / "merge_summary.json",
        {
            "base_model": args.base_model,
            "adapter_path": str(args.adapter_path),
            "output_dir": str(args.output_dir),
        },
    )
    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

