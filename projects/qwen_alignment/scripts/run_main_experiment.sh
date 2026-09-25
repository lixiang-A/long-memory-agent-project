#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

source "$PROJECT_DIR/scripts/net_env.sh"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"

"$PYTHON" -m alignment_course.prepare_hh \
  --dataset-name Anthropic/hh-rlhf \
  --source-rows 2000 \
  --train-size 1600 \
  --val-size 200 \
  --test-size 200 \
  --output-dir data/processed

"$PYTHON" -m alignment_course.train_sft \
  --base-model "$BASE_MODEL" \
  --train-path data/processed/hh_train.jsonl \
  --eval-path data/processed/hh_val.jsonl \
  --output-dir artifacts/main/sft_adapter \
  --max-steps 120 \
  --max-seq-length 1024

"$PYTHON" -m alignment_course.merge_lora \
  --base-model "$BASE_MODEL" \
  --adapter-path artifacts/main/sft_adapter/final \
  --output-dir artifacts/main/sft_merged

"$PYTHON" -m alignment_course.train_dpo \
  --model-name-or-path artifacts/main/sft_merged \
  --train-path data/processed/hh_train.jsonl \
  --eval-path data/processed/hh_val.jsonl \
  --output-dir artifacts/main/dpo_adapter \
  --max-steps 200 \
  --max-seq-length 1024 \
  --max-prompt-length 512

"$PYTHON" -m alignment_course.train_reward_model \
  --base-model "$BASE_MODEL" \
  --train-path data/processed/hh_train.jsonl \
  --eval-path data/processed/hh_val.jsonl \
  --output-dir artifacts/main/reward_adapter \
  --max-steps 200 \
  --max-seq-length 1024

"$PYTHON" -m alignment_course.train_ppo_pilot \
  --policy-model artifacts/main/sft_merged \
  --reward-model artifacts/main/reward_adapter/final \
  --reward-base "$BASE_MODEL" \
  --train-path data/processed/hh_train.jsonl \
  --output-dir artifacts/main/ppo_pilot \
  --max-prompts 128 \
  --total-episodes 128 \
  --response-length 64 || true

"$PYTHON" -m alignment_course.evaluate \
  --test-path data/processed/hh_test.jsonl \
  --reference-model artifacts/main/sft_merged \
  --dpo-model artifacts/main/dpo_adapter/final \
  --dpo-base artifacts/main/sft_merged \
  --ppo-model artifacts/main/ppo_pilot/final \
  --ppo-base artifacts/main/sft_merged \
  --reward-model artifacts/main/reward_adapter/final \
  --reward-base "$BASE_MODEL" \
  --output-json artifacts/main/eval_metrics.json \
  --max-pairs 200 \
  --num-generations 50 \
  --max-seq-length 1024 \
  --max-new-tokens 128

"$PYTHON" -m alignment_course.plot_metrics \
  --artifact-dir artifacts/main \
  --output-dir artifacts/main/figures
