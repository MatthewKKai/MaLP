#!/bin/bash
# =============================================================================
# MaLP - Stage 4: LoRA Fine-Tuning
# =============================================================================
# This script fine-tunes the knowledge-adapted LLaMA model using LoRA
# on user-specific dialogue data.
#
# Training settings (Section 4.1):
#   - Optimizer: AdamW
#   - Learning rate: 5e-5
#   - Warm-up: 10% of total steps
#   - Weight decay: 1e-4
#   - LoRA rank: 8, alpha: 32
#   - Max input length: 1024, max output length: 2048
#
# Prerequisites:
#   - Knowledge-adapted model available (from Stage 2)
#   - Fine-tuning data prepared (from Stage 3)
#
# Usage:
#   # Single GPU
#   bash scripts/run_finetune.sh
#
#   # Multi-GPU (set NUM_GPUS)
#   NUM_GPUS=2 bash scripts/run_finetune.sh
# =============================================================================

set -e

# Configuration
MODEL_PATH="./pretrained_model/base_model"  # Output from knowledge injection
DATA_PATH="./training_data/finetune_data.json"
VAL_DATA_PATH="./training_data/finetune_data_val.json"
OUTPUT_DIR="./finetuned_model"
EPOCHS=1
BATCH_SIZE=1
LEARNING_RATE=5e-5
WEIGHT_DECAY=1e-4
WARMUP_RATIO=0.1
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.05
MAX_INPUT_LENGTH=1024
MAX_OUTPUT_LENGTH=2048
NUM_GPUS=${NUM_GPUS:-1}

echo "============================================"
echo "MaLP - LoRA Fine-Tuning Stage"
echo "============================================"
echo "Model path: ${MODEL_PATH}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "LoRA rank: ${LORA_R}, alpha: ${LORA_ALPHA}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "============================================"

if [ ${NUM_GPUS} -gt 1 ]; then
    echo "Using Distributed Data Parallel with ${NUM_GPUS} GPUs"
    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --data_path ${DATA_PATH} \
        --val_data_path ${VAL_DATA_PATH} \
        --model_path ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --max_input_length ${MAX_INPUT_LENGTH} \
        --max_output_length ${MAX_OUTPUT_LENGTH} \
        --use_ddp \
        --fp16
else
    echo "Using single GPU training"
    python train.py \
        --data_path ${DATA_PATH} \
        --val_data_path ${VAL_DATA_PATH} \
        --model_path ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --max_input_length ${MAX_INPUT_LENGTH} \
        --max_output_length ${MAX_OUTPUT_LENGTH} \
        --fp16
fi

echo "Fine-tuning complete! Model saved to ${OUTPUT_DIR}"
