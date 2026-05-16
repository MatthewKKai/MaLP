"""
MaLP Inference Stage - Personalized Response Generation.

Implements Equation 3 from the paper:
    p = Retriever(x)
    x, p -> Φ_hat -> y

Where:
- x is the new user query
- p is the prompt retrieved from memory M
- Φ_hat is the LoRA-tuned LLM
- y is the personalized response

Usage:
    # Interactive mode
    python inference.py \
        --model_path ./finetuned_model \
        --memory_path ./memory_output \
        --interactive

    # Batch mode
    python inference.py \
        --model_path ./finetuned_model \
        --memory_path ./memory_output \
        --query "What should I do about my recurring headaches?" \
        --dialogue_history ../data/dialogues2_cleaned.json
"""

import argparse
import json
import logging
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.memory import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MaLPInference:
    """MaLP inference pipeline combining memory retrieval and LoRA-tuned LLM.

    Args:
        model_path (str): Path to the LoRA fine-tuned model.
        base_model_path (str, optional): Path to the base model (if separate).
        memory_path (str, optional): Path to the memory output directory.
        device (str): Device to run on. Default: "auto".
    """

    def __init__(self, model_path: str, base_model_path: str = None,
                 memory_path: str = None, device: str = "auto"):
        self.device = self._get_device(device)

        # Load model
        logger.info(f"Loading model from {model_path}")
        self._load_model(model_path, base_model_path)

        # Load memory
        self.memory = None
        if memory_path and os.path.exists(memory_path):
            self._load_memory(memory_path)

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self, model_path: str, base_model_path: str = None):
        """Load the LoRA fine-tuned model."""
        try:
            # Try loading as a PEFT model
            if base_model_path:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                self.model = PeftModel.from_pretrained(base_model, model_path)
            else:
                # Try loading directly (model saved with save_pretrained)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load as PEFT model: {e}")
            logger.info("Attempting to load as standard model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def _load_memory(self, memory_path: str):
        """Load formed memory from disk."""
        self.memory = Memory()

        # Load STM
        stm_path = os.path.join(memory_path, "stm.json")
        if os.path.exists(stm_path):
            with open(stm_path, "r", encoding="utf-8") as f:
                stm_data = json.load(f)
            for item in stm_data:
                self.memory.add_to_stm(item["key"], item["value"])
            logger.info(f"Loaded {len(stm_data)} STM entries")

        # Load LTM
        ltm_path = os.path.join(memory_path, "ltm.json")
        if os.path.exists(ltm_path):
            with open(ltm_path, "r", encoding="utf-8") as f:
                ltm_data = json.load(f)
            for item in ltm_data:
                self.memory.add_to_ltm(item["key"], item["value"])
            logger.info(f"Loaded {len(ltm_data)} LTM entries")

    def retrieve_memory(self, query: str) -> str:
        """Retrieve relevant knowledge from memory.

        Implements: p = Retriever(x)

        Args:
            query: The user's new query.

        Returns:
            Retrieved memory prompt string.
        """
        if self.memory is None:
            return ""
        return self.memory.retrieve(query)

    def generate(self, query: str, dialogue_history: str = "",
                 max_length: int = 512, temperature: float = 0.7,
                 top_p: float = 0.95, top_k: int = 50) -> str:
        """Generate a personalized response using the full MaLP pipeline.

        Implements: x, p -> Φ_hat -> y

        Args:
            query: The user's new query (x).
            dialogue_history: Previous dialogue context.
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.

        Returns:
            Generated personalized response (y).
        """
        # Step 1: Retrieve memory prompt (p = Retriever(x))
        memory_prompt = self.retrieve_memory(query)

        # Step 2: Construct full input
        input_parts = []
        if memory_prompt:
            input_parts.append(f"Relevant knowledge:\n{memory_prompt}\n")
        if dialogue_history:
            input_parts.append(f"Previous dialogue:\n{dialogue_history}\n")
        input_parts.append(f"Patient: {query}\nDoctor:")

        full_input = "\n".join(input_parts)

        # Step 3: Generate response (x, p -> Φ_hat -> y)
        input_ids = self.tokenizer(
            full_input, return_tensors="pt", truncation=True, max_length=1024
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (exclude input)
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()


def interactive_mode(pipeline: MaLPInference):
    """Run interactive inference mode."""
    print("\n" + "=" * 60)
    print("MaLP - Personalized Medical Assistant")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    dialogue_history = []

    while True:
        try:
            query = input("Patient: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ["quit", "exit", "q"]:
            break

        if not query:
            continue

        # Format history
        history_str = ""
        if dialogue_history:
            history_str = "\n".join(dialogue_history[-6:])  # Last 3 turns

        # Generate response
        response = pipeline.generate(query, dialogue_history=history_str)
        print(f"Doctor: {response}\n")

        # Update history
        dialogue_history.append(f"Patient: {query}")
        dialogue_history.append(f"Doctor: {response}")


def main():
    parser = argparse.ArgumentParser(
        description="MaLP Inference - Personalized Response Generation"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base_model_path", type=str, default=None,
        help="Path to the base model (if LoRA adapter is separate)"
    )
    parser.add_argument(
        "--memory_path", type=str, default=None,
        help="Path to the memory output directory"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query for batch mode"
    )
    parser.add_argument(
        "--dialogue_history", type=str, default=None,
        help="Path to dialogue history JSON for context"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Maximum generation length (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (default: auto)"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = MaLPInference(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        memory_path=args.memory_path,
        device=args.device,
    )

    if args.interactive:
        interactive_mode(pipeline)
    elif args.query:
        # Single query mode
        history_str = ""
        if args.dialogue_history and os.path.exists(args.dialogue_history):
            with open(args.dialogue_history, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            if isinstance(history_data, list) and len(history_data) > 0:
                # Use first dialogue as history
                dialogue = history_data[0]
                parts = []
                for round_data in dialogue:
                    if isinstance(round_data, dict):
                        for key, value in round_data.items():
                            if isinstance(value, dict):
                                parts.append(f"Patient: {value.get('User', '')}")
                                parts.append(f"Doctor: {value.get('Assistant', '')}")
                history_str = "\n".join(parts)

        response = pipeline.generate(
            args.query,
            dialogue_history=history_str,
            max_length=args.max_length,
            temperature=args.temperature,
        )
        print(f"\nQuery: {args.query}")
        print(f"Response: {response}")
    else:
        print("Please specify --interactive or --query. Use --help for options.")


if __name__ == "__main__":
    main()
