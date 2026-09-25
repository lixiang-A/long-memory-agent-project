#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"

# Source network proxy settings if needed
source "$PROJECT_DIR/scripts/net_env.sh" 2>/dev/null || true

echo "============================================"
echo "DPO Extended Training + Auto-Evaluation"
echo "============================================"
echo ""

# ── Run 1: DPO 600 steps, beta=0.1 ──
RUN_NAME="dpo_600steps"
OUT_DIR="artifacts/main/${RUN_NAME}"

echo "[1/2] Training DPO (600 steps, beta=0.1)..."
echo "      Output: ${OUT_DIR}"
echo "      Estimated time: ~2.5 hours on M3"
echo ""

"$PYTHON" -m alignment_course.train_dpo \
  --model-name-or-path artifacts/main/sft_merged \
  --train-path data/processed/hh_train.jsonl \
  --eval-path data/processed/hh_val.jsonl \
  --output-dir "${OUT_DIR}" \
  --max-steps 600 \
  --beta 0.1 \
  --max-seq-length 1024 \
  --max-prompt-length 512

echo ""
echo "[2/2] Evaluating DPO 600-step model..."
echo ""

"$PYTHON" -m alignment_course.evaluate \
  --test-path data/processed/hh_test.jsonl \
  --reference-model artifacts/main/sft_merged \
  --dpo-model "${OUT_DIR}/final" \
  --dpo-base artifacts/main/sft_merged \
  --reward-model artifacts/main/reward_adapter/final \
  --reward-base Qwen/Qwen3-0.6B \
  --output-json "${OUT_DIR}/eval_metrics.json" \
  --max-pairs 200 \
  --num-generations 50 \
  --max-seq-length 1024 \
  --max-new-tokens 128

echo ""
echo "============================================"
echo "Done!"
echo "Results: ${OUT_DIR}/eval_metrics.json"
echo "============================================"

# Quick summary
"$PYTHON" -c "
import json
with open('${OUT_DIR}/eval_metrics.json') as f:
    d = json.load(f)
print(f'DPO 600-step preference accuracy: {d[\"dpo_preference\"][\"accuracy\"]:.4f}')
print(f'DPO approx KL to SFT:          {d[\"dpo_approx_kl_to_reference\"]:.4f}')
print(f'Runtime: {d[\"runtime_seconds\"]:.0f}s')
print()
print(f'SFT baseline (for comparison): 0.5500')
print(f'DPO 200-step (for comparison): 0.5500')
"
