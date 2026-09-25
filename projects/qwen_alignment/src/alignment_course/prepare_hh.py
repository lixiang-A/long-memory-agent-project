from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from alignment_course.data_prep import parse_hh_pair, parsed_to_row
from alignment_course.io_utils import save_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Anthropic HH-RLHF preference data.")
    parser.add_argument("--dataset-name", default="Anthropic/hh-rlhf")
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-rows", type=int, default=2000)
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    needed = args.train_size + args.val_size + args.test_size
    dataset = load_dataset(args.dataset_name, split=f"{args.split}[:{args.source_rows}]")

    rows = []
    parse_failures = 0
    for idx, item in enumerate(dataset):
        parsed = parse_hh_pair(idx, item.get("chosen", ""), item.get("rejected", ""))
        if parsed is None:
            parse_failures += 1
            continue
        rows.append(parsed_to_row(parsed))

    if len(rows) < needed:
        raise RuntimeError(
            f"Only {len(rows)} usable rows after parsing; {needed} are required for the split."
        )

    train_rows = rows[: args.train_size]
    val_rows = rows[args.train_size : args.train_size + args.val_size]
    test_rows = rows[args.train_size + args.val_size : needed]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "hh_train.jsonl", train_rows)
    write_jsonl(args.output_dir / "hh_val.jsonl", val_rows)
    write_jsonl(args.output_dir / "hh_test.jsonl", test_rows)
    save_json(
        args.output_dir / "summary.json",
        {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "source_rows_requested": args.source_rows,
            "source_rows_loaded": len(dataset),
            "usable_rows": len(rows),
            "parse_failures": parse_failures,
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "test_size": len(test_rows),
            "fields": ["source_idx", "prompt", "chosen", "rejected"],
        },
    )
    print(f"Wrote HH-RLHF preference splits to {args.output_dir}")
    print(f"usable_rows={len(rows)} parse_failures={parse_failures}")


if __name__ == "__main__":
    main()

