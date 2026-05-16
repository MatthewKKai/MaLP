#!/bin/bash
# =============================================================================
# MaLP - Stage 3: Data Preparation
# =============================================================================
# This script prepares training data for both the knowledge injection and
# LoRA fine-tuning stages.
#
# Modes:
#   - knowledge: Extracts medical Q&A pairs for domain adaptation
#   - finetune: Creates memory-augmented training examples for LoRA
#
# Prerequisites:
#   - Dialogue data available in ../data/ directory
#   - For finetune mode: memory formation stage must be completed first
#
# Usage:
#   bash scripts/run_prepare_data.sh
# =============================================================================

set -e

DIALOGUE_PATH="../data/dialogues2_cleaned.json"
MEMORY_PATH="./memory_output"
OUTPUT_DIR="./training_data"

mkdir -p ${OUTPUT_DIR}

echo "============================================"
echo "MaLP - Data Preparation Stage"
echo "============================================"

# Step 1: Prepare knowledge injection data
echo ""
echo "Step 1: Preparing knowledge injection data..."
python prepare_data.py \
    --dialogue_path ${DIALOGUE_PATH} \
    --output_path ${OUTPUT_DIR}/knowledge_data.json \
    --mode knowledge

# Step 2: Prepare fine-tuning data (with memory)
echo ""
echo "Step 2: Preparing fine-tuning data with memory..."
python prepare_data.py \
    --dialogue_path ${DIALOGUE_PATH} \
    --memory_path ${MEMORY_PATH} \
    --output_path ${OUTPUT_DIR}/finetune_data.json \
    --mode finetune

echo ""
echo "Data preparation complete! Files saved to ${OUTPUT_DIR}"
echo "  - ${OUTPUT_DIR}/knowledge_data.json (for knowledge injection)"
echo "  - ${OUTPUT_DIR}/finetune_data.json (for LoRA fine-tuning)"
echo "  - ${OUTPUT_DIR}/finetune_data_val.json (validation set)"
