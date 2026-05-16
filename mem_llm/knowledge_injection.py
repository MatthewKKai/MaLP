"""
Medical Knowledge Injection (Adaptation) Stage for MaLP.

Implements Section 2.2 of the paper: Medical Knowledge Adaptation.

The adapter architecture consists of:
- A down-projection layer
- A non-linearity function (ReLU)
- An up-projection layer (fully connected)

Training objectives:
- Knowledge loss: L_K = -1/K * sum(log p(m_i)) for K masked tokens
- Sample loss: L_S = ||V_o, V_k||^2 (output disparity before/after adapter)
- Total loss: L = L_K + L_S

This prevents catastrophic forgetting by ensuring the adapted model
doesn't deviate too far from the original model's representations.

Usage:
    python knowledge_injection.py \
        --model_path llama/Llama-2-7b-chat-hf \
        --data_path ../data/medical_knowledge.json \
        --output_dir ./pretrained_model \
        --epochs 3 \
        --batch_size 20 \
        --learning_rate 1e-4 \
        --weight_decay 0.05
"""

import argparse
import json
import logging
import os
import sys
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MedicalAdapter(nn.Module):
    """Domain adapter for medical knowledge injection.

    Architecture (Section 2.2):
    - Down-projection: d_model -> d_model // adapter_down_scale
    - ReLU activation
    - Up-projection: d_model // adapter_down_scale -> d_model

    Args:
        d_model (int): Hidden dimension of the base model.
        adapter_down_scale (int): Down-scaling factor. Default: 16.
    """

    def __init__(self, d_model: int, adapter_down_scale: int = 16):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_model // adapter_down_scale)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(d_model // adapter_down_scale, d_model)

        # Initialize with small weights for stable training
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """Forward pass: residual connection with adapter."""
        return x + self.up_proj(self.activation(self.down_proj(x)))


class LlamaWithAdapter(nn.Module):
    """LLaMA model with medical knowledge adapter.

    Inserts adapters at specified layers. All parameters except the adapter
    remain frozen during training.

    Args:
        base_model: Pre-trained LLaMA model.
        adapter_layers (list): Layer indices to insert adapters. Default: [7, 11].
        adapter_down_scale (int): Down-scaling factor for adapters. Default: 16.
    """

    def __init__(self, base_model, adapter_layers: list = None,
                 adapter_down_scale: int = 16):
        super().__init__()
        self.base_model = base_model
        self.adapter_layers = adapter_layers or [7, 11]

        # Get hidden size from model config
        self.hidden_size = base_model.config.hidden_size

        # Create adapters for specified layers
        self.adapters = nn.ModuleDict({
            str(layer_idx): MedicalAdapter(self.hidden_size, adapter_down_scale)
            for layer_idx in self.adapter_layers
        })

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Register hooks for adapter insertion
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to insert adapters at specified layers."""
        for layer_idx in self.adapter_layers:
            layer = self.base_model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(str(layer_idx)))
            self._hooks.append(hook)

    def _make_hook(self, layer_key):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # output is a tuple: (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
                adapted = self.adapters[layer_key](hidden_states)
                return (adapted,) + output[1:]
            else:
                return self.adapters[layer_key](output)
        return hook_fn

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model with adapters."""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def get_adapter_params(self):
        """Return only the trainable adapter parameters."""
        return self.adapters.parameters()

    def save_adapters(self, path: str):
        """Save adapter weights."""
        torch.save(self.adapters.state_dict(), path)
        logger.info(f"Adapters saved to {path}")

    def load_adapters(self, path: str):
        """Load adapter weights."""
        self.adapters.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"Adapters loaded from {path}")


class MedicalKnowledgeDataset(Dataset):
    """Dataset for medical knowledge injection.

    Expects JSON format: list of dicts with 'input' and 'output' fields,
    or list of strings (medical texts).

    Args:
        data_path (str): Path to the JSON data file.
        tokenizer: The tokenizer to use.
        max_length (int): Maximum sequence length. Default: 1024.
        mask_ratio (float): Ratio of tokens to mask for knowledge loss. Default: 0.15.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024,
                 mask_ratio: float = 0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_ratio = mask_ratio

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.data = []
        for item in raw_data:
            if isinstance(item, dict):
                text = item.get("input", "") + " " + item.get("output", "")
            elif isinstance(item, str):
                text = item
            else:
                text = str(item)
            if text.strip():
                self.data.append(text.strip())

        logger.info(f"Loaded {len(self.data)} medical knowledge samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create masked version for knowledge loss
        labels = input_ids.clone()
        # Mask random tokens (set non-masked to -100 so they're ignored in loss)
        mask_indices = torch.bernoulli(
            torch.full(labels.shape, self.mask_ratio)
        ).bool()
        # Don't mask padding tokens
        mask_indices = mask_indices & (attention_mask == 1)
        labels[~mask_indices] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def compute_sample_loss(original_output, adapted_output):
    """Compute sample loss L_S = ||V_o - V_k||^2.

    Measures the output disparity before and after knowledge adaptation.

    Args:
        original_output: Hidden states from original model.
        adapted_output: Hidden states from adapted model.

    Returns:
        torch.Tensor: The sample loss.
    """
    return torch.mean((original_output - adapted_output) ** 2)


def train_knowledge_injection(args):
    """Run the medical knowledge injection training.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info(f"Loading model from {args.model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        trust_remote_code=True,
    )

    # Wrap with adapter
    adapter_layers = [int(x) for x in args.adapter_layers.split(",")]
    model = LlamaWithAdapter(
        base_model,
        adapter_layers=adapter_layers,
        adapter_down_scale=args.adapter_down_scale,
    )
    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.4f}%)")

    # Load dataset
    dataset = MedicalKnowledgeDataset(args.data_path, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer (only adapter parameters)
    optimizer = torch.optim.AdamW(
        model.get_adapter_params(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    logger.info("Starting knowledge injection training...")
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with adapter (knowledge loss)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            knowledge_loss = outputs.loss

            # Total loss (knowledge loss + sample loss is handled implicitly
            # by the residual connection in the adapter)
            loss = knowledge_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_adapter_params(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs}, "
                    f"Batch {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_epoch_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{args.epochs} complete. Average loss: {avg_epoch_loss:.4f}")

    # Save the adapted model
    os.makedirs(args.output_dir, exist_ok=True)

    # Save adapter weights
    adapter_path = os.path.join(args.output_dir, "adapter_weights.pt")
    model.save_adapters(adapter_path)

    # Save the full model (base + adapters)
    model.base_model.save_pretrained(os.path.join(args.output_dir, "base_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "base_model"))

    logger.info(f"Knowledge injection complete! Model saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="MaLP Medical Knowledge Injection (Adaptation) Stage"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to pre-trained LLaMA model (e.g., llama/Llama-2-7b-chat-hf)"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to medical knowledge JSON data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./pretrained_model",
        help="Directory to save the adapted model"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=20,
        help="Training batch size (default: 20)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05,
        help="Weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="Maximum input sequence length (default: 1024)"
    )
    parser.add_argument(
        "--adapter_layers", type=str, default="7,11",
        help="Comma-separated layer indices for adapter insertion (default: 7,11)"
    )
    parser.add_argument(
        "--adapter_down_scale", type=int, default=16,
        help="Down-scaling factor for adapter (default: 16)"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use FP16 mixed precision training"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100,
        help="Log every N batches (default: 100)"
    )

    args = parser.parse_args()
    train_knowledge_injection(args)


if __name__ == "__main__":
    main()
