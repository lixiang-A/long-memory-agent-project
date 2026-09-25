from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from alignment_course.io_utils import ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot available trainer logs and evaluation metrics.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_trainer_state(path: Path) -> List[Dict[str, Any]]:
    state_path = path / "trainer_state.json"
    if not state_path.exists():
        return []
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return state.get("log_history", [])


def plot_log_history(method: str, log_history: List[Dict[str, Any]], output_dir: Path) -> None:
    if not log_history:
        return
    loss_points = [(item.get("step"), item.get("loss")) for item in log_history if "loss" in item]
    eval_points = [(item.get("step"), item.get("eval_loss")) for item in log_history if "eval_loss" in item]
    if not loss_points and not eval_points:
        return
    plt.figure(figsize=(7, 4))
    if loss_points:
        x, y = zip(*loss_points)
        plt.plot(x, y, marker="o", label="train loss")
    if eval_points:
        x, y = zip(*eval_points)
        plt.plot(x, y, marker="s", label="eval loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"{method} training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{method}_loss.png", dpi=180)
    plt.close()


def plot_eval_metrics(artifact_dir: Path, output_dir: Path) -> None:
    eval_path = artifact_dir / "eval_metrics.json"
    if not eval_path.exists():
        return
    metrics = load_json(eval_path)
    labels = []
    values = []
    dpo_pref = metrics.get("dpo_preference", {})
    if "accuracy" in dpo_pref:
        labels.append("DPO pref acc")
        values.append(dpo_pref["accuracy"])
    ppo_pref = metrics.get("ppo_preference", {})
    if "accuracy" in ppo_pref:
        labels.append("PPO pref acc")
        values.append(ppo_pref["accuracy"])
    win = metrics.get("rm_win_rate", {})
    if win.get("status") == "completed":
        labels.extend(["DPO RM win", "PPO RM win"])
        values.extend([win["dpo_win_rate"], win["ppo_win_rate"]])
    if not labels:
        return
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color=["#3568a8", "#b34e4a", "#2b8a5e", "#805ca8"][: len(labels)])
    plt.ylim(0, 1)
    plt.ylabel("rate")
    plt.title("Evaluation summary")
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_summary.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    for method, rel in [
        ("sft", "sft_adapter"),
        ("dpo", "dpo_adapter"),
        ("reward", "reward_adapter"),
        ("ppo_pilot", "ppo_pilot"),
    ]:
        plot_log_history(method, load_trainer_state(args.artifact_dir / rel), args.output_dir)
    plot_eval_metrics(args.artifact_dir, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()

