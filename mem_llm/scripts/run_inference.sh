#!/bin/bash
# =============================================================================
# MaLP - Stage 6: Inference (Personalized Response Generation)
# =============================================================================
# This script runs the full MaLP inference pipeline:
#   p = Retriever(x)      -> Retrieve knowledge from memory
#   x, p -> Φ_hat -> y    -> Generate personalized response
#
# Prerequisites:
#   - Fine-tuned model available (from Stage 4)
#   - Memory formed (from Stage 1)
#
# Usage:
#   # Interactive mode
#   bash scripts/run_inference.sh --interactive
#
#   # Single query mode
#   bash scripts/run_inference.sh --query "What should I do about my headaches?"
# =============================================================================

set -e

# Configuration
MODEL_PATH="./finetuned_model"
BASE_MODEL_PATH="./pretrained_model/base_model"
MEMORY_PATH="./memory_output"
DIALOGUE_HISTORY="../data/dialogues2_cleaned.json"
MAX_LENGTH=512
TEMPERATURE=0.7

echo "============================================"
echo "MaLP - Inference (Personalized Response)"
echo "============================================"

# Parse arguments
EXTRA_ARGS=""
for arg in "$@"; do
    EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
done

if [ -z "${EXTRA_ARGS}" ]; then
    # Default: interactive mode
    EXTRA_ARGS="--interactive"
fi

python inference.py \
    --model_path ${MODEL_PATH} \
    --base_model_path ${BASE_MODEL_PATH} \
    --memory_path ${MEMORY_PATH} \
    --dialogue_history ${DIALOGUE_HISTORY} \
    --max_length ${MAX_LENGTH} \
    --temperature ${TEMPERATURE} \
    ${EXTRA_ARGS}
