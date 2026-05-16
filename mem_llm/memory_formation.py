"""
Memory Formation Stage for MaLP.

Implements the DPeM (Dual-Process enhanced Memory) mechanism as described
in Section 2.3 and 2.4.1 of the paper:

1. Rehearsal Process:
   - Learning: Pass dialogues d_i to coordinator C to extract notes (nt_s = C(d_i))
   - Summarizing: Filter relevant notes, categorize into knowledge types

2. Executive Process:
   - Memorizing: Store knowledge in STM with type labels
   - Transit: Move frequently accessed knowledge from STM to LTM (threshold θ)

Usage:
    python memory_formation.py \
        --dialogue_path ../data/dialogues2_cleaned.json \
        --output_dir ./memory_output \
        --transit_threshold 3 \
        --stm_refresh_interval 5 \
        --model gpt-4.1-mini
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.memory import Memory
from model.utils import ChatGPTWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Prompts for the Coordinator (C) in the DPeM mechanism
LEARNING_PROMPT = """You are a medical knowledge coordinator. Given the following dialogue between a patient and a doctor, please extract all important notes including:
1. Patient's personal information (age, conditions, history)
2. Patient's symptoms and concerns
3. Medical advice given by the doctor
4. Patient's dialogue preferences (concise, detailed, polite)

Dialogue:
{dialogue}

Please list each note as a separate item, one per line. Format each note as:
- [type] content

Where type is one of: personal_info, symptom, medical_advice, preference, common_sense
"""

SUMMARIZING_PROMPT = """You are a medical knowledge coordinator. Given the following notes extracted from a patient-doctor dialogue, please:
1. Filter out irrelevant or redundant notes
2. Categorize each remaining note as either "common-sense" or "user-specific" knowledge
3. Return only the relevant, categorized knowledge items

Notes:
{notes}

Please return the filtered knowledge in the following JSON format:
[
    {{"type": "common-sense" or "user-specific", "key": "brief description", "value": "detailed knowledge content"}},
    ...
]

Return ONLY the JSON array, no other text.
"""


def format_dialogue(dialogue_rounds: list) -> str:
    """Format a list of dialogue rounds into a readable string.

    Args:
        dialogue_rounds: List of round dicts, each containing 'Assistant' and 'User' keys.

    Returns:
        Formatted dialogue string.
    """
    formatted = []
    for i, round_data in enumerate(dialogue_rounds):
        # Handle different data formats
        if isinstance(round_data, dict):
            # Format: {round_idx: {"Assistant": ..., "User": ...}}
            for key, value in round_data.items():
                if isinstance(value, dict):
                    assistant = value.get("Assistant", "")
                    user = value.get("User", "")
                    formatted.append(f"Round {i}:")
                    formatted.append(f"  Patient: {user}")
                    formatted.append(f"  Doctor: {assistant}")
                elif key == "Assistant":
                    formatted.append(f"  Doctor: {value}")
                elif key == "User":
                    formatted.append(f"  Patient: {value}")
    return "\n".join(formatted)


def learning_step(coordinator: ChatGPTWrapper, dialogue: str) -> str:
    """Rehearsal Process - Learning Step.

    Pass dialogue d_i to coordinator C to extract notes: nt_s = C(d_i)

    Args:
        coordinator: The ChatGPT-based coordinator.
        dialogue: Formatted dialogue string.

    Returns:
        Extracted notes string.
    """
    prompt = LEARNING_PROMPT.format(dialogue=dialogue)
    messages = [{"role": "user", "content": prompt}]
    notes = coordinator.obtain_answer(messages)
    return notes


def summarizing_step(coordinator: ChatGPTWrapper, notes: str) -> list:
    """Rehearsal Process - Summarizing Step.

    Filter and categorize notes into knowledge items.

    Args:
        coordinator: The ChatGPT-based coordinator.
        notes: Raw notes from the learning step.

    Returns:
        List of knowledge dicts with 'type', 'key', 'value' fields.
    """
    prompt = SUMMARIZING_PROMPT.format(notes=notes)
    messages = [{"role": "user", "content": prompt}]
    response = coordinator.obtain_answer(messages)

    # Parse the JSON response
    try:
        # Try to extract JSON from the response
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code block
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        knowledge_items = json.loads(response)
        if isinstance(knowledge_items, list):
            return knowledge_items
    except json.JSONDecodeError:
        logger.warning("Failed to parse coordinator response as JSON. Attempting fallback parsing.")
        # Fallback: try to extract items manually
        return _fallback_parse(response)

    return []


def _fallback_parse(response: str) -> list:
    """Fallback parser for when JSON parsing fails."""
    items = []
    for line in response.strip().split("\n"):
        line = line.strip().strip("-").strip()
        if not line:
            continue
        if "common-sense" in line.lower() or "common sense" in line.lower():
            items.append({
                "type": "common-sense",
                "key": line[:50],
                "value": line,
            })
        elif "user-specific" in line.lower() or "user specific" in line.lower():
            items.append({
                "type": "user-specific",
                "key": line[:50],
                "value": line,
            })
        else:
            items.append({
                "type": "common-sense",
                "key": line[:50],
                "value": line,
            })
    return items


def executive_process(memory: Memory, knowledge_items: list):
    """Executive Process - Memorizing Step.

    Store categorized knowledge in STM and handle transit to LTM.

    Args:
        memory: The Memory instance.
        knowledge_items: List of knowledge dicts from summarizing step.
    """
    for item in knowledge_items:
        k_type = item.get("type", "common-sense")
        key = item.get("key", "")
        value = item.get("value", "")

        if key and value:
            # Store in STM with type-prefixed key: k_type : k_i
            stm_key = f"{k_type}: {key}"
            memory.add_to_stm(stm_key, value)
            logger.debug(f"Added to STM: {stm_key}")


def run_memory_formation(args):
    """Run the complete DPeM memory formation pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Initialize coordinator (C)
    coordinator = ChatGPTWrapper(model=args.model)
    logger.info(f"Coordinator initialized with model: {args.model}")

    # Initialize Memory (M)
    memory = Memory(
        transit_threshold=args.transit_threshold,
        ltm_model_name=args.ltm_model,
    )
    logger.info(f"Memory initialized with transit_threshold={args.transit_threshold}")

    # Load dialogues
    with open(args.dialogue_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    logger.info(f"Loaded {len(dialogues)} dialogues from {args.dialogue_path}")

    # Process each dialogue through DPeM
    total_dialogues = min(len(dialogues), args.max_dialogues) if args.max_dialogues > 0 else len(dialogues)

    for i in range(total_dialogues):
        dialogue_data = dialogues[i]
        logger.info(f"Processing dialogue {i+1}/{total_dialogues}")

        # Format dialogue
        if isinstance(dialogue_data, list):
            dialogue_text = format_dialogue(dialogue_data)
        elif isinstance(dialogue_data, str):
            dialogue_text = dialogue_data
        else:
            dialogue_text = str(dialogue_data)

        if not dialogue_text.strip():
            logger.warning(f"Skipping empty dialogue {i}")
            continue

        # === Rehearsal Process ===

        # Step 1: Learning - Extract notes from dialogue
        notes = learning_step(coordinator, dialogue_text)
        if not notes:
            logger.warning(f"No notes extracted from dialogue {i}")
            continue

        # Store notes in working memory
        memory.add_to_working_memory(notes)

        # Step 2: Summarizing - Filter and categorize knowledge
        knowledge_items = summarizing_step(coordinator, notes)
        logger.info(f"  Extracted {len(knowledge_items)} knowledge items")

        # === Executive Process ===

        # Step 3: Memorizing - Store in STM
        executive_process(memory, knowledge_items)

        # Refresh working memory after each iteration
        memory.refresh_working_memory()

        # Periodically refresh STM (triggers transit to LTM)
        if (i + 1) % args.stm_refresh_interval == 0:
            logger.info(f"  Refreshing STM at iteration {i+1} (transit check)")
            memory.transit()  # Check for STM -> LTM transit before refresh
            # Note: We don't fully clear STM here, just transit high-freq items
            # Full refresh happens at larger intervals as per paper

        # Full STM refresh at larger intervals
        if (i + 1) % (args.stm_refresh_interval * 5) == 0:
            logger.info(f"  Full STM refresh at iteration {i+1}")
            memory.refresh_stm()

    # Final transit check
    memory.transit()

    # Save memory state
    os.makedirs(args.output_dir, exist_ok=True)
    memory_state = memory.get_memory_state()

    # Save STM
    stm_data = []
    for key, value in memory_state["stm"]["items"]:
        stm_data.append({"key": key, "value": value})
    with open(os.path.join(args.output_dir, "stm.json"), "w", encoding="utf-8") as f:
        json.dump(stm_data, f, indent=2, ensure_ascii=False)

    # Save LTM
    ltm_data = []
    for key, value in memory_state["ltm"]["items"]:
        ltm_data.append({"key": key, "value": value})
    with open(os.path.join(args.output_dir, "ltm.json"), "w", encoding="utf-8") as f:
        json.dump(ltm_data, f, indent=2, ensure_ascii=False)

    # Save full memory state summary
    summary = {
        "total_dialogues_processed": total_dialogues,
        "stm_size": memory_state["stm"]["size"],
        "ltm_size": memory_state["ltm"]["size"],
        "frequency_table": memory_state["stm"]["frequency_table"],
    }
    with open(os.path.join(args.output_dir, "memory_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\nMemory formation complete!")
    logger.info(f"  STM entries: {memory_state['stm']['size']}")
    logger.info(f"  LTM entries: {memory_state['ltm']['size']}")
    logger.info(f"  Output saved to: {args.output_dir}")

    return memory


def main():
    parser = argparse.ArgumentParser(
        description="MaLP Memory Formation Stage - DPeM Pipeline"
    )
    parser.add_argument(
        "--dialogue_path", type=str, required=True,
        help="Path to the dialogue JSON file (e.g., ../data/dialogues2_cleaned.json)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./memory_output",
        help="Directory to save the formed memory"
    )
    parser.add_argument(
        "--transit_threshold", type=int, default=3,
        help="Frequency threshold θ for STM -> LTM transit (default: 3)"
    )
    parser.add_argument(
        "--stm_refresh_interval", type=int, default=5,
        help="Number of iterations between STM transit checks (default: 5)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4.1-mini",
        help="LLM model for the coordinator C (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--ltm_model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence transformer model for LTM embeddings (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--max_dialogues", type=int, default=0,
        help="Maximum number of dialogues to process (0 = all, default: 0)"
    )

    args = parser.parse_args()
    run_memory_formation(args)


if __name__ == "__main__":
    main()
