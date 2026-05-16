"""
Data Preparation Script for MaLP.

Converts the raw dialogue data (from data/ directory) into the training
format expected by the fine-tuning stage (train.py).

The training format is a JSON list of dicts with 'input' and 'output' fields:
- input: The dialogue context + memory prompt + new query
- output: The expected personalized response

This script also supports preparing data for the knowledge injection stage
by extracting medical Q&A pairs.

Usage:
    # Prepare fine-tuning data (with memory)
    python prepare_data.py \
        --dialogue_path ../data/dialogues2_cleaned.json \
        --memory_path ./memory_output \
        --output_path ./training_data/finetune_data.json \
        --mode finetune

    # Prepare knowledge injection data
    python prepare_data.py \
        --dialogue_path ../data/dialogues2_cleaned.json \
        --output_path ./training_data/knowledge_data.json \
        --mode knowledge
"""

import argparse
import json
import logging
import os
import sys
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_memory(memory_path: str) -> dict:
    """Load formed memory from the memory formation stage.

    Args:
        memory_path: Path to the memory output directory.

    Returns:
        Dict with 'stm' and 'ltm' lists of knowledge items.
    """
    memory = {"stm": [], "ltm": []}

    stm_path = os.path.join(memory_path, "stm.json")
    if os.path.exists(stm_path):
        with open(stm_path, "r", encoding="utf-8") as f:
            memory["stm"] = json.load(f)
        logger.info(f"Loaded {len(memory['stm'])} STM items")

    ltm_path = os.path.join(memory_path, "ltm.json")
    if os.path.exists(ltm_path):
        with open(ltm_path, "r", encoding="utf-8") as f:
            memory["ltm"] = json.load(f)
        logger.info(f"Loaded {len(memory['ltm'])} LTM items")

    return memory


def format_memory_prompt(memory: dict, max_items: int = 5) -> str:
    """Format memory items into a prompt string for the model.

    Args:
        memory: Dict with 'stm' and 'ltm' lists.
        max_items: Maximum number of memory items to include.

    Returns:
        Formatted memory prompt string.
    """
    prompt_parts = []

    # Include LTM items (higher priority as they are frequently accessed)
    ltm_items = memory.get("ltm", [])[:max_items]
    if ltm_items:
        prompt_parts.append("Relevant knowledge from long-term memory:")
        for item in ltm_items:
            prompt_parts.append(f"  - {item.get('key', '')}: {item.get('value', '')}")

    # Include STM items
    remaining = max_items - len(ltm_items)
    stm_items = memory.get("stm", [])[:remaining]
    if stm_items:
        prompt_parts.append("Recent knowledge from short-term memory:")
        for item in stm_items:
            prompt_parts.append(f"  - {item.get('key', '')}: {item.get('value', '')}")

    return "\n".join(prompt_parts)


def prepare_finetune_data(dialogues: list, memory: dict, output_path: str,
                          train_ratio: float = 0.9):
    """Prepare fine-tuning data with memory-augmented prompts.

    For each dialogue, creates training examples where:
    - input = dialogue history + memory prompt + current query
    - output = the doctor's response

    Args:
        dialogues: List of dialogue data.
        memory: Loaded memory dict.
        output_path: Path to save the training data.
        train_ratio: Train/validation split ratio.
    """
    training_examples = []
    memory_prompt = format_memory_prompt(memory)

    for dialogue_idx, dialogue in enumerate(dialogues):
        if not isinstance(dialogue, list):
            continue

        # Build training examples from each dialogue
        history = []
        for round_idx, round_data in enumerate(dialogue):
            if isinstance(round_data, dict):
                for key, value in round_data.items():
                    if isinstance(value, dict):
                        user_msg = value.get("User", "").strip().strip('"')
                        assistant_msg = value.get("Assistant", "").strip().strip('"')

                        if user_msg and assistant_msg:
                            # Create training example with context
                            context = ""
                            if history:
                                context = "Previous dialogue:\n"
                                for h in history[-3:]:  # Last 3 turns for context
                                    context += f"Patient: {h['user']}\nDoctor: {h['assistant']}\n"

                            # Construct input with memory prompt
                            input_text = ""
                            if memory_prompt:
                                input_text += f"{memory_prompt}\n\n"
                            if context:
                                input_text += f"{context}\n"
                            input_text += f"Patient: {user_msg}\nDoctor:"

                            training_examples.append({
                                "input": input_text,
                                "output": assistant_msg,
                            })

                            history.append({
                                "user": user_msg,
                                "assistant": assistant_msg,
                            })

    # Shuffle and split
    random.shuffle(training_examples)
    split_idx = int(len(training_examples) * train_ratio)
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    val_path = output_path.replace(".json", "_val.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Prepared {len(train_data)} training examples -> {output_path}")
    logger.info(f"Prepared {len(val_data)} validation examples -> {val_path}")


def prepare_knowledge_data(dialogues: list, output_path: str):
    """Prepare data for the knowledge injection stage.

    Extracts medical Q&A pairs from dialogues for domain adaptation.

    Args:
        dialogues: List of dialogue data.
        output_path: Path to save the knowledge data.
    """
    knowledge_examples = []

    for dialogue in dialogues:
        if not isinstance(dialogue, list):
            continue

        for round_data in dialogue:
            if isinstance(round_data, dict):
                for key, value in round_data.items():
                    if isinstance(value, dict):
                        user_msg = value.get("User", "").strip().strip('"')
                        assistant_msg = value.get("Assistant", "").strip().strip('"')

                        if user_msg and assistant_msg and len(assistant_msg) > 20:
                            knowledge_examples.append({
                                "input": user_msg,
                                "output": assistant_msg,
                            })

    # Shuffle
    random.shuffle(knowledge_examples)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_examples, f, indent=2, ensure_ascii=False)

    logger.info(f"Prepared {len(knowledge_examples)} knowledge examples -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MaLP Data Preparation"
    )
    parser.add_argument(
        "--dialogue_path", type=str, required=True,
        help="Path to dialogue JSON file"
    )
    parser.add_argument(
        "--memory_path", type=str, default=None,
        help="Path to memory output directory (for finetune mode)"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the prepared data"
    )
    parser.add_argument(
        "--mode", type=str, choices=["finetune", "knowledge"], required=True,
        help="Preparation mode: 'finetune' for LoRA fine-tuning, 'knowledge' for injection"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9,
        help="Train/validation split ratio (default: 0.9)"
    )

    args = parser.parse_args()

    # Load dialogues
    with open(args.dialogue_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    logger.info(f"Loaded {len(dialogues)} dialogues from {args.dialogue_path}")

    if args.mode == "finetune":
        memory = {}
        if args.memory_path and os.path.exists(args.memory_path):
            memory = load_memory(args.memory_path)
        else:
            logger.warning("No memory path provided or path doesn't exist. "
                         "Preparing data without memory prompts.")
        prepare_finetune_data(dialogues, memory, args.output_path, args.train_ratio)

    elif args.mode == "knowledge":
        prepare_knowledge_data(dialogues, args.output_path)


if __name__ == "__main__":
    main()
