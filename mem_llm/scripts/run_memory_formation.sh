#!/bin/bash
# =============================================================================
# MaLP - Stage 1: Memory Formation (DPeM Pipeline)
# =============================================================================
# This script runs the Dual-Process enhanced Memory (DPeM) formation stage.
# It processes historical dialogues through the coordinator (C) to extract,
# categorize, and store knowledge in STM and LTM.
#
# Prerequisites:
#   - Set OPENAI_API_KEY environment variable
#   - Dialogue data available in ../data/ directory
#
# Usage:
#   bash scripts/run_memory_formation.sh
# =============================================================================

set -e

# Configuration
DIALOGUE_PATH="../data/dialogues2_cleaned.json"
OUTPUT_DIR="./memory_output"
TRANSIT_THRESHOLD=3
STM_REFRESH_INTERVAL=5
MODEL="gpt-4.1-mini"
MAX_DIALOGUES=0  # 0 = process all dialogues

echo "============================================"
echo "MaLP - Memory Formation Stage"
echo "============================================"
echo "Dialogue path: ${DIALOGUE_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Transit threshold: ${TRANSIT_THRESHOLD}"
echo "Model: ${MODEL}"
echo "============================================"

python memory_formation.py \
    --dialogue_path ${DIALOGUE_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --transit_threshold ${TRANSIT_THRESHOLD} \
    --stm_refresh_interval ${STM_REFRESH_INTERVAL} \
    --model ${MODEL} \
    --max_dialogues ${MAX_DIALOGUES}

echo "Memory formation complete! Output saved to ${OUTPUT_DIR}"
