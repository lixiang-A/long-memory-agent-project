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
  --source-rows 80 \
  --train-size 40 \
  --val-size 10 \
  --test-size 10 \
  --output-dir data/processed_smoke

"$PYTHON" -m alignment_course.train_sft \
  --base-model "$BASE_MODEL" \
  --train-path data/processed_smoke/hh_train.jsonl \
  --eval-path data/processed_smoke/hh_val.jsonl \
  --output-dir artifacts/smoke/sft_adapter \
  --max-train-samples 20 \
  --max-eval-samples 10 \
  --max-steps 2 \
  --max-seq-length 512

"$PYTHON" -m alignment_course.merge_lora \
  --base-model "$BASE_MODEL" \
  --adapter-path artifacts/smoke/sft_adapter/final \
  --output-dir artifacts/smoke/sft_merged

"$PYTHON" -m alignment_course.train_dpo \
  --model-name-or-path artifacts/smoke/sft_merged \
  --train-path data/processed_smoke/hh_train.jsonl \
  --eval-path data/processed_smoke/hh_val.jsonl \
  --output-dir artifacts/smoke/dpo_adapter \
  --max-train-samples 20 \
  --max-eval-samples 10 \
  --max-steps 2 \
  --max-seq-length 512 \
  --max-prompt-length 256

"$PYTHON" -m alignment_course.train_reward_model \
  --base-model "$BASE_MODEL" \
  --train-path data/processed_smoke/hh_train.jsonl \
  --eval-path data/processed_smoke/hh_val.jsonl \
  --output-dir artifacts/smoke/reward_adapter \
  --max-train-samples 20 \
  --max-eval-samples 10 \
  --max-steps 2 \
  --max-seq-length 512

"$PYTHON" -m alignment_course.evaluate \
  --test-path data/processed_smoke/hh_test.jsonl \
  --reference-model artifacts/smoke/sft_merged \
  --dpo-model artifacts/smoke/dpo_adapter/final \
  --dpo-base artifacts/smoke/sft_merged \
  --reward-model artifacts/smoke/reward_adapter/final \
  --reward-base "$BASE_MODEL" \
  --output-json artifacts/smoke/eval_metrics.json \
  --max-pairs 10 \
  --num-generations 4 \
  --max-seq-length 512 \
  --max-new-tokens 64
