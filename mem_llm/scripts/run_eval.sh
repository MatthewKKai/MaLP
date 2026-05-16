#!/bin/bash
# =============================================================================
# MaLP - Stage 5: Evaluation
# =============================================================================
# This script runs the evaluation pipeline on three tasks (Section 4.3):
#   1. Question Answering (QA) - Profile QA and Knowledge QA (ROUGE-1, ROUGE-L)
#   2. Preference Classification - Accuracy
#   3. Response Generation - Win Rate
#
# Prerequisites:
#   - Fine-tuned model available (from Stage 4)
#   - Memory formed (from Stage 1)
#   - Set OPENAI_API_KEY environment variable
#
# Usage:
#   bash scripts/run_eval.sh
# =============================================================================

set -e

# Configuration
MODEL_PATH="./finetuned_model"
MEMORY_PATH="./memory_output"
DIALOGUE_PATH="../data/dialogues2_cleaned.json"
PROFILES_PATH="../dialogue_generation/profiles_4.json"
OUTPUT_DIR="./eval_results"
TASK="all"  # Options: all, qa, preference, response
NUM_SAMPLES=100
MODEL="gpt-4.1-mini"

echo "============================================"
echo "MaLP - Evaluation Stage"
echo "============================================"
echo "Model path: ${MODEL_PATH}"
echo "Memory path: ${MEMORY_PATH}"
echo "Task: ${TASK}"
echo "Num samples: ${NUM_SAMPLES}"
echo "============================================"

python eval.py \
    --model_path ${MODEL_PATH} \
    --memory_path ${MEMORY_PATH} \
    --dialogue_path ${DIALOGUE_PATH} \
    --profiles_path ${PROFILES_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --task ${TASK} \
    --num_samples ${NUM_SAMPLES} \
    --model ${MODEL}

echo "Evaluation complete! Results saved to ${OUTPUT_DIR}"
