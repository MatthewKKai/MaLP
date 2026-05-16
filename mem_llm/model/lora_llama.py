"""
LoRA-wrapped LLaMA model for MaLP fine-tuning.

This module provides a wrapper class that applies Low-Rank Adaptation (LoRA)
to a pre-trained LLaMA model following the paper's specifications:
- LoRA rank: 8
- Scaling factor (alpha): 32
- Target modules: query and value projection layers
- Task type: Causal Language Modeling
"""

from peft import get_peft_model, LoraConfig, TaskType


class lora_llama:
    """Wrapper class that applies LoRA configuration to a LLaMA model.

    Args:
        base_model: A pre-trained LLaMA model (e.g., LlamaForCausalLM).
        lora_r (int): Rank of the LoRA update matrices. Default: 8.
        lora_alpha (int): Scaling factor for LoRA. Default: 32.
        lora_dropout (float): Dropout probability for LoRA layers. Default: 0.05.
        target_modules (list): List of module names to apply LoRA to.
            Default: ["q_proj", "v_proj"].
    """

    def __init__(self, base_model, lora_r: int = 8, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, target_modules: list = None):
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]

        # Define LoRA configuration as per the paper:
        # rank=8, alpha=32, applied to query and value projections
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
        )

        # Apply LoRA to the base model
        self.model = get_peft_model(self.base_model, self.peft_config)

    def get_lora_llama(self):
        """Return the LoRA-wrapped model."""
        return self.model
