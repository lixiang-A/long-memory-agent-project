#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen/Qwen3-0.6B}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/llm_large10k}"
DATA_DIR="$WORK_DIR/data/processed_10k"
ARTIFACT_DIR="$WORK_DIR/artifacts/large10k"

mkdir -p "$DATA_DIR" "$ARTIFACT_DIR"

python -m alignment_course.prepare_hh \
  --dataset-name Anthropic/hh-rlhf \
  --source-rows 10000 \
  --train-size 8000 \
  --val-size 1000 \
  --test-size 1000 \
  --output-dir "$DATA_DIR"

python -m alignment_course.train_sft \
  --base-model "$BASE_MODEL" \
  --train-path "$DATA_DIR/hh_train.jsonl" \
  --eval-path "$DATA_DIR/hh_val.jsonl" \
  --output-dir "$ARTIFACT_DIR/sft_adapter" \
  --max-steps 500 \
  --max-seq-length 1024

python -m alignment_course.merge_lora \
  --base-model "$BASE_MODEL" \
  --adapter-path "$ARTIFACT_DIR/sft_adapter/final" \
  --output-dir "$ARTIFACT_DIR/sft_merged"

python -m alignment_course.train_dpo \
  --model-name-or-path "$ARTIFACT_DIR/sft_merged" \
  --train-path "$DATA_DIR/hh_train.jsonl" \
  --eval-path "$DATA_DIR/hh_val.jsonl" \
  --output-dir "$ARTIFACT_DIR/dpo_beta0.1_steps1000" \
  --max-steps 1000 \
  --beta 0.1 \
  --max-seq-length 1024 \
  --max-prompt-length 512

python -m alignment_course.evaluate \
  --test-path "$DATA_DIR/hh_test.jsonl" \
  --reference-model "$ARTIFACT_DIR/sft_merged" \
  --dpo-model "$ARTIFACT_DIR/dpo_beta0.1_steps1000/final" \
  --dpo-base "$ARTIFACT_DIR/sft_merged" \
  --reward-model artifacts/main/reward_adapter/final \
  --reward-base "$BASE_MODEL" \
  --output-json "$ARTIFACT_DIR/eval_dpo_beta0.1_steps1000.json" \
  --max-pairs 1000 \
  --num-generations 0
