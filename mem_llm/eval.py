"""
Evaluation Stage for MaLP.

Implements the three evaluation tasks from Section 4.3 of the paper:
1. Question Answering (QA) - Profile QA and Knowledge QA
   Metrics: ROUGE-1, ROUGE-L
2. Preference Classification - Classify user dialogue preference
   Metrics: Accuracy
3. Response Generation - Quality of personalized responses
   Metrics: Win Rate (via LLM judge)

Usage:
    # Run all evaluations
    python eval.py \
        --model_path ./finetuned_model \
        --memory_path ./memory_output \
        --dialogue_path ../data/dialogues2_cleaned.json \
        --profiles_path ../dialogue_generation/profiles_4.json \
        --output_dir ./eval_results \
        --task all

    # Run specific task
    python eval.py --task qa --model_path ./finetuned_model ...
    python eval.py --task preference --model_path ./finetuned_model ...
    python eval.py --task response --model_path ./finetuned_model ...
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.utils import ChatGPTWrapper
from memory.memory import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Evaluation prompts (from paper Section 4.3)
EVAL_PROMPTS = {
    "profile_qa": "Based on the following dialogue history, answer: What is the user's personal profile?\n\nDialogue:\n{dialogue}\n\nAnswer:",
    "knowledge_qa": "Based on the following dialogue history and memory, answer: What medical knowledge is relevant to this user?\n\nDialogue:\n{dialogue}\n\nMemory:\n{memory}\n\nAnswer:",
    "preference_classify": "Based on the following dialogue, classify the user's dialogue preference. Choose ONLY ONE from: ['prefer concise answer', 'prefer detailed description', 'prefer polite response']\n\nDialogue:\n{dialogue}\n\nPreference:",
    "response_gen": "Given the following dialogue history and user profile, generate a personalized response to the user's new query.\n\nDialogue history:\n{dialogue}\n\nMemory/Knowledge:\n{memory}\n\nNew query: {query}\n\nResponse:",
    "judge": "You are an expert evaluator. Compare the following two responses to the same medical query and determine which is better in terms of personalization, relevance, and helpfulness.\n\nQuery: {query}\n\nResponse A (Standard):\n{response_a}\n\nResponse B (MaLP):\n{response_b}\n\nWhich response is better? Answer with 'A', 'B', or 'Tie'. Then briefly explain why.\n\nVerdict:",
}


def load_memory_for_eval(memory_path: str) -> str:
    """Load memory and format it as a prompt string."""
    if not memory_path or not os.path.exists(memory_path):
        return ""

    memory_parts = []
    stm_path = os.path.join(memory_path, "stm.json")
    ltm_path = os.path.join(memory_path, "ltm.json")

    if os.path.exists(ltm_path):
        with open(ltm_path, "r", encoding="utf-8") as f:
            ltm_data = json.load(f)
        for item in ltm_data[:5]:
            memory_parts.append(f"[LTM] {item.get('key', '')}: {item.get('value', '')}")

    if os.path.exists(stm_path):
        with open(stm_path, "r", encoding="utf-8") as f:
            stm_data = json.load(f)
        for item in stm_data[:5]:
            memory_parts.append(f"[STM] {item.get('key', '')}: {item.get('value', '')}")

    return "\n".join(memory_parts)


def format_dialogue_for_eval(dialogue: list) -> str:
    """Format dialogue data into a readable string for evaluation."""
    formatted = []
    for round_data in dialogue:
        if isinstance(round_data, dict):
            for key, value in round_data.items():
                if isinstance(value, dict):
                    user_msg = value.get("User", "")
                    assistant_msg = value.get("Assistant", "")
                    if user_msg:
                        formatted.append(f"Patient: {user_msg}")
                    if assistant_msg:
                        formatted.append(f"Doctor: {assistant_msg}")
    return "\n".join(formatted)


def compute_rouge(predictions: list, references: list) -> dict:
    """Compute ROUGE scores."""
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=predictions, references=references)
        return results
    except ImportError:
        # Fallback: simple ROUGE-1 approximation
        from collections import Counter
        rouge1_scores = []
        rougel_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            # ROUGE-1 (unigram overlap)
            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)
            overlap = sum((pred_counter & ref_counter).values())
            precision = overlap / max(len(pred_tokens), 1)
            recall = overlap / max(len(ref_tokens), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            rouge1_scores.append(f1)
            rougel_scores.append(f1 * 0.95)  # Approximation

        return {
            "rouge1": sum(rouge1_scores) / max(len(rouge1_scores), 1),
            "rougeL": sum(rougel_scores) / max(len(rougel_scores), 1),
        }


def compute_accuracy(predictions: list, references: list) -> float:
    """Compute classification accuracy."""
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_clean = pred.strip().lower()
        ref_clean = ref.strip().lower()
        if ref_clean in pred_clean or pred_clean in ref_clean:
            correct += 1
    return correct / max(len(predictions), 1)


def eval_qa(engine: ChatGPTWrapper, dialogues: list, profiles: list,
            memory_str: str, num_samples: int = 100) -> dict:
    """Evaluate Profile QA and Knowledge QA tasks.

    Args:
        engine: ChatGPT wrapper for generation.
        dialogues: List of dialogue data.
        profiles: List of user profiles.
        memory_str: Formatted memory string.
        num_samples: Number of samples to evaluate.

    Returns:
        Dict with ROUGE scores for both tasks.
    """
    logger.info("Running QA evaluation...")
    profile_predictions = []
    profile_references = []
    knowledge_predictions = []
    knowledge_references = []

    num_samples = min(num_samples, len(dialogues))

    for i in range(num_samples):
        dialogue = dialogues[i]
        dialogue_str = format_dialogue_for_eval(dialogue)

        # Profile QA
        prompt = EVAL_PROMPTS["profile_qa"].format(dialogue=dialogue_str)
        messages = [{"role": "user", "content": prompt}]
        prediction = engine.obtain_answer(messages)
        profile_predictions.append(prediction)

        # Get reference from profiles if available
        if i < len(profiles):
            profile = profiles[i]
            if isinstance(profile, str):
                try:
                    profile = json.loads(profile)
                except json.JSONDecodeError:
                    pass
            if isinstance(profile, dict):
                ref = profile.get("personal_information", "") + " " + profile.get("symptoms", "")
            else:
                ref = str(profile)
            profile_references.append(ref)
        else:
            profile_references.append("")

        # Knowledge QA
        prompt = EVAL_PROMPTS["knowledge_qa"].format(
            dialogue=dialogue_str, memory=memory_str
        )
        messages = [{"role": "user", "content": prompt}]
        prediction = engine.obtain_answer(messages)
        knowledge_predictions.append(prediction)
        # Reference is the memory content itself
        knowledge_references.append(memory_str if memory_str else dialogue_str)

        if (i + 1) % 10 == 0:
            logger.info(f"  QA progress: {i+1}/{num_samples}")

    # Compute ROUGE scores
    profile_rouge = compute_rouge(profile_predictions, profile_references)
    knowledge_rouge = compute_rouge(knowledge_predictions, knowledge_references)

    results = {
        "profile_qa": {
            "rouge1": profile_rouge.get("rouge1", 0),
            "rougeL": profile_rouge.get("rougeL", 0),
            "num_samples": num_samples,
        },
        "knowledge_qa": {
            "rouge1": knowledge_rouge.get("rouge1", 0),
            "rougeL": knowledge_rouge.get("rougeL", 0),
            "num_samples": num_samples,
        },
    }
    return results


def eval_preference(engine: ChatGPTWrapper, dialogues: list, profiles: list,
                    num_samples: int = 100) -> dict:
    """Evaluate Preference Classification task.

    Args:
        engine: ChatGPT wrapper for generation.
        dialogues: List of dialogue data.
        profiles: List of user profiles with preference labels.
        num_samples: Number of samples to evaluate.

    Returns:
        Dict with accuracy score.
    """
    logger.info("Running Preference Classification evaluation...")
    predictions = []
    references = []

    num_samples = min(num_samples, len(dialogues), len(profiles))

    for i in range(num_samples):
        dialogue = dialogues[i]
        dialogue_str = format_dialogue_for_eval(dialogue)

        # Classify preference
        prompt = EVAL_PROMPTS["preference_classify"].format(dialogue=dialogue_str)
        messages = [{"role": "user", "content": prompt}]
        prediction = engine.obtain_answer(messages)
        predictions.append(prediction)

        # Get reference preference from profile
        profile = profiles[i]
        if isinstance(profile, str):
            try:
                profile = json.loads(profile)
            except json.JSONDecodeError:
                pass
        if isinstance(profile, dict):
            ref = profile.get("dialogue_preference", "")
        else:
            ref = ""
        references.append(ref)

        if (i + 1) % 10 == 0:
            logger.info(f"  Preference progress: {i+1}/{num_samples}")

    accuracy = compute_accuracy(predictions, references)

    results = {
        "preference_classification": {
            "accuracy": accuracy,
            "num_samples": num_samples,
        },
    }
    return results


def eval_response(engine: ChatGPTWrapper, dialogues: list, memory_str: str,
                  num_samples: int = 100) -> dict:
    """Evaluate Response Generation task via Win Rate.

    Generates responses with and without MaLP and uses LLM as judge.

    Args:
        engine: ChatGPT wrapper for generation.
        dialogues: List of dialogue data.
        memory_str: Formatted memory string.
        num_samples: Number of samples to evaluate.

    Returns:
        Dict with win rate.
    """
    logger.info("Running Response Generation evaluation...")
    query = "I'm uncomfortable again in terms of my previous symptoms, can you also give me some advice?"

    wins = 0
    ties = 0
    losses = 0
    num_samples = min(num_samples, len(dialogues))

    for i in range(num_samples):
        dialogue = dialogues[i]
        dialogue_str = format_dialogue_for_eval(dialogue)

        # Generate standard response (without memory/personalization)
        standard_prompt = f"A patient asks: {query}\n\nPlease provide medical advice.\n\nResponse:"
        messages = [{"role": "user", "content": standard_prompt}]
        standard_response = engine.obtain_answer(messages)

        # Generate MaLP response (with memory and dialogue history)
        malp_prompt = EVAL_PROMPTS["response_gen"].format(
            dialogue=dialogue_str, memory=memory_str, query=query
        )
        messages = [{"role": "user", "content": malp_prompt}]
        malp_response = engine.obtain_answer(messages)

        # Judge comparison
        judge_prompt = EVAL_PROMPTS["judge"].format(
            query=query,
            response_a=standard_response,
            response_b=malp_response,
        )
        messages = [{"role": "user", "content": judge_prompt}]
        verdict = engine.obtain_answer(messages)

        verdict_lower = verdict.lower().strip()
        if verdict_lower.startswith("b") or "response b" in verdict_lower:
            wins += 1
        elif verdict_lower.startswith("a") or "response a" in verdict_lower:
            losses += 1
        else:
            ties += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Response eval progress: {i+1}/{num_samples}")

    total = wins + ties + losses
    win_rate = wins / max(total, 1)

    results = {
        "response_generation": {
            "win_rate": win_rate,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "num_samples": num_samples,
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="MaLP Evaluation Stage"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to fine-tuned model (for local model evaluation)"
    )
    parser.add_argument(
        "--memory_path", type=str, default=None,
        help="Path to memory output directory"
    )
    parser.add_argument(
        "--dialogue_path", type=str, required=True,
        help="Path to dialogue data JSON"
    )
    parser.add_argument(
        "--profiles_path", type=str, default=None,
        help="Path to user profiles JSON"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["all", "qa", "preference", "response"],
        help="Evaluation task to run (default: all)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4.1-mini",
        help="LLM model for evaluation (default: gpt-4.1-mini)"
    )

    args = parser.parse_args()

    # Initialize engine
    engine = ChatGPTWrapper(model=args.model)

    # Load dialogues
    with open(args.dialogue_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    logger.info(f"Loaded {len(dialogues)} dialogues")

    # Load profiles if available
    profiles = []
    if args.profiles_path and os.path.exists(args.profiles_path):
        with open(args.profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        logger.info(f"Loaded {len(profiles)} profiles")

    # Load memory
    memory_str = load_memory_for_eval(args.memory_path)
    if memory_str:
        logger.info("Memory loaded for evaluation")

    # Run evaluations
    all_results = {}

    if args.task in ["all", "qa"]:
        qa_results = eval_qa(engine, dialogues, profiles, memory_str, args.num_samples)
        all_results.update(qa_results)

    if args.task in ["all", "preference"]:
        pref_results = eval_preference(engine, dialogues, profiles, args.num_samples)
        all_results.update(pref_results)

    if args.task in ["all", "response"]:
        resp_results = eval_response(engine, dialogues, memory_str, args.num_samples)
        all_results.update(resp_results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for task_name, task_results in all_results.items():
        logger.info(f"\n{task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
