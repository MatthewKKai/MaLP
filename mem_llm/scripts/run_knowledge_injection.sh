#!/bin/bash
# =============================================================================
# MaLP - Stage 2: Medical Knowledge Injection (Domain Adaptation)
# =============================================================================
# This script runs the medical knowledge adaptation stage which injects
# domain-specific medical knowledge into the base LLM via adapters.
#
# The adapter architecture (Section 2.2):
#   - Down-projection layer
#   - ReLU activation
#   - Up-projection layer
#
# Training settings (Section 4.1):
#   - Learning rate: 1e-4
#   - Batch size: 20
#   - Weight decay: 0.05
#
# Prerequisites:
#   - Pre-trained LLaMA model available
#   - Medical knowledge data prepared (run prepare_data.py --mode knowledge first)
#
# Usage:
#   bash scripts/run_knowledge_injection.sh
# =============================================================================

set -e

# Configuration
MODEL_PATH="llama/Llama-2-7b-chat-hf"
DATA_PATH="./training_data/knowledge_data.json"
OUTPUT_DIR="./pretrained_model"
EPOCHS=3
BATCH_SIZE=20
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.05
MAX_LENGTH=1024
ADAPTER_LAYERS="7,11"

echo "============================================"
echo "MaLP - Knowledge Injection Stage"
echo "============================================"
echo "Model path: ${MODEL_PATH}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "============================================"

python knowledge_injection.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --max_length ${MAX_LENGTH} \
    --adapter_layers ${ADAPTER_LAYERS} \
    --fp16

echo "Knowledge injection complete! Model saved to ${OUTPUT_DIR}"
