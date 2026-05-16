"""
LoRA Fine-Tuning Stage for MaLP.

Implements Section 2.4.2 of the paper: Memory Utilization via PEFT.

Fine-tunes a LLaMA model using LoRA (Low-Rank Adaptation) on user-specific
dialogue data to learn user preferences and generate personalized responses.

Training settings (from paper Section 4.1):
- Optimizer: AdamW
- Learning rate: 5e-5
- Warm-up: 10% of total steps
- Weight decay: 1e-4
- LoRA rank: 8, alpha: 32
- Max input length: 1024, max output length: 2048

Usage:
    # Single GPU
    python train.py \
        --data_path ./training_data/finetune_data.json \
        --model_path llama/Llama-2-7b-chat-hf \
        --output_dir ./finetuned_model \
        --epochs 1 \
        --batch_size 1 \
        --learning_rate 5e-5

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 train.py \
        --data_path ./training_data/finetune_data.json \
        --model_path llama/Llama-2-7b-chat-hf \
        --output_dir ./finetuned_model \
        --epochs 1 \
        --batch_size 1 \
        --learning_rate 5e-5 \
        --use_ddp
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.lora_llama import lora_llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DialogueDataset(Dataset):
    """Dataset for LoRA fine-tuning on dialogue data.

    Expects JSON format: list of dicts with 'input' and 'output' fields.
    - input: dialogue context + memory prompt + query
    - output: expected personalized response

    Args:
        data: List of dialogue dicts.
        tokenizer: The tokenizer to use.
        max_input_length (int): Maximum input sequence length. Default: 1024.
        max_output_length (int): Maximum output sequence length. Default: 2048.
    """

    def __init__(self, data: list, tokenizer, max_input_length: int = 1024,
                 max_output_length: int = 2048):
        self.labels = []
        self.input_ids = []
        self.attn_masks = []

        for item in data:
            input_text = item.get("input", "")
            output_text = item.get("output", "")

            if not input_text or not output_text:
                continue

            # Encode input
            encoding_input = tokenizer(
                input_text,
                truncation=True,
                max_length=max_input_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.input_ids.append(encoding_input["input_ids"].squeeze(0))
            self.attn_masks.append(encoding_input["attention_mask"].squeeze(0))

            # Encode output (labels)
            encoding_output = tokenizer(
                output_text,
                truncation=True,
                max_length=max_output_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.labels.append(encoding_output["input_ids"].squeeze(0))

        logger.info(f"Loaded {len(self.input_ids)} training examples")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


def data_collator(data):
    """Custom data collator for the Trainer."""
    return {
        "input_ids": torch.stack([f[0] for f in data]),
        "attention_mask": torch.stack([f[1] for f in data]),
        "labels": torch.stack([f[2] for f in data]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="MaLP LoRA Fine-Tuning Stage"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to training data JSON (output of prepare_data.py)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to pre-trained LLaMA model"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./finetuned_model",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Per-device training batch size (default: 1)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
        help="Warmup ratio (default: 0.1)"
    )
    parser.add_argument(
        "--max_input_length", type=int, default=1024,
        help="Maximum input sequence length (default: 1024)"
    )
    parser.add_argument(
        "--max_output_length", type=int, default=2048,
        help="Maximum output sequence length (default: 2048)"
    )
    parser.add_argument(
        "--lora_r", type=int, default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32,
        help="LoRA alpha scaling factor (default: 32)"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout (default: 0.05)"
    )
    parser.add_argument(
        "--use_ddp", action="store_true",
        help="Use Distributed Data Parallel (multi-GPU)"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100,
        help="Log every N steps (default: 100)"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use FP16 mixed precision"
    )
    parser.add_argument(
        "--val_data_path", type=str, default=None,
        help="Path to validation data JSON (optional)"
    )

    args = parser.parse_args()

    # Handle DDP initialization
    if args.use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    logger.info(f"Using device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    try:
        base_model = LlamaForCausalLM.from_pretrained(args.model_path)
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )

    tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    logger.info(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_model = lora_llama(
        base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).get_lora_llama()
    lora_model.print_trainable_parameters()

    # Load training data
    logger.info(f"Loading training data from {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Create dataset
    dataset = DialogueDataset(
        train_data, tokenizer, args.max_input_length, args.max_output_length
    )

    # Split into train/val if no separate val data
    if args.val_data_path and os.path.exists(args.val_data_path):
        with open(args.val_data_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        val_dataset = DialogueDataset(
            val_data, tokenizer, args.max_input_length, args.max_output_length
        )
        train_dataset = dataset
    else:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Training arguments (as per paper Section 4.1)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=3,
        fp16=args.fp16,
        report_to="none",
        ddp_find_unused_parameters=False if args.use_ddp else None,
    )

    # Initialize Trainer
    if args.use_ddp:
        lora_model = lora_model.to(device)
        ddp_model = DDP(lora_model, device_ids=[local_rank])
        trainer = Trainer(
            model=ddp_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
    else:
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

    # Train
    logger.info("Starting LoRA fine-tuning...")
    torch.cuda.empty_cache()
    trainer.train()

    # Save the fine-tuned model
    logger.info(f"Saving fine-tuned model to {args.output_dir}")
    if args.use_ddp:
        if local_rank == 0:
            lora_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
    else:
        lora_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    logger.info("Fine-tuning complete!")

    # Quick test generation
    if local_rank == 0:
        test_query = "How to treat fever?"
        logger.info(f"Test generation with query: {test_query}")
        input_ids = tokenizer(test_query, return_tensors="pt").input_ids.to(device)
        model_for_gen = lora_model if not args.use_ddp else lora_model
        model_for_gen.eval()
        with torch.no_grad():
            outputs = model_for_gen.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                max_length=512,
                top_p=0.95,
                temperature=0.7,
            )
        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.info(f"Generated response: {generated[0][:200]}...")


if __name__ == "__main__":
    main()
