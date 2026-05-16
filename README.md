# MaLP: Memory-Augmented LLM Personalization with Short- and Long-Term Memory Coordination

[![Paper](https://img.shields.io/badge/Paper-NAACL%202024-blue)](https://aclanthology.org/2024.naacl-long.132/)

This repository contains the implementation of **MaLP** (Memory-augmented LLM Personalization), a framework that integrates a novel **Dual-Process enhanced Memory (DPeM)** mechanism with **Parameter-Efficient Fine-Tuning (PEFT)** to personalize medical assistants based on user-specific needs and dialogue history.

## Overview

MaLP addresses two key challenges in personalized LLM-based medical assistants:

1. **Memory Structure**: Instead of simple dictionary-based memory, MaLP implements a biologically-inspired memory mechanism with three types of memory (Working Memory, Short-Term Memory, Long-Term Memory) coordinated through a dual-process schema.

2. **Efficient Personalization**: Rather than fully retraining an LLM, MaLP uses LoRA (Low-Rank Adaptation) to fine-tune the model on user-specific dialogues with minimal computational cost.

### Architecture

```
Historical Dialogues → Coordinator (C) → DPeM Memory Formation
                                              ↓
                    Working Memory → STM → LTM (via transit threshold θ)
                                              ↓
New Query → Retriever(x) → Memory Prompt (p)
                                              ↓
                    x, p → LoRA-tuned LLM (Φ̂) → Personalized Response (y)
```

## Dataset

We first derive patient's profile from public medical corpus ([Medical-Dialogue-System](https://github.com/UCSD-AI4H/Medical-Dialogue-System)) and then endow the patient's profile to a powerful chat model. Assistant role (e.g., doctor) will be simulated independently using the same chat model and thus we could collect the historical dialogues via self-chat between these two roles.

![data_collection](https://github.com/user-attachments/assets/8b520849-44ee-4c47-b427-2c234254133e)

### Data Structure

- **Profile** - Contains personal information, symptoms, and dialogue preference
- **Dialogue** - Contains multi-round dialogues generated via self-chat

## Project Structure

```
MaLP/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   ├── dialogues2_cleaned.json        # Cleaned dialogue dataset
│   └── dialogues_3_cleaned.json       # Additional dialogue data
├── dialogue_generation/
│   ├── dialogue_generation.py         # Self-chat dialogue generation
│   ├── profile_creation.py            # Patient profile generation
│   ├── profiles_4.json                # Generated patient profiles
│   ├── prompts.py                     # Prompts for generation
│   └── utils.py                       # ChatGPT wrapper utilities
└── mem_llm/
    ├── memory/
    │   ├── __init__.py                # Memory module exports
    │   ├── memory.py                  # Unified Memory (M) coordinator
    │   ├── dynamic_memory.py          # STM (Levenshtein retrieval)
    │   ├── static_memory.py           # LTM (semantic retrieval)
    │   └── prompts.py                 # Memory/evaluation prompts
    ├── model/
    │   ├── __init__.py                # Model module exports
    │   ├── lora_llama.py              # LoRA wrapper for LLaMA
    │   └── utils.py                   # ChatGPT wrapper for coordinator
    ├── scripts/
    │   ├── run_memory_formation.sh    # Stage 1 script
    │   ├── run_knowledge_injection.sh # Stage 2 script
    │   ├── run_prepare_data.sh        # Stage 3 script
    │   ├── run_finetune.sh            # Stage 4 script
    │   ├── run_eval.sh                # Stage 5 script
    │   └── run_inference.sh           # Stage 6 script
    ├── memory_formation.py            # DPeM memory formation pipeline
    ├── knowledge_injection.py         # Medical knowledge adaptation
    ├── prepare_data.py                # Data preparation utilities
    ├── train.py                       # LoRA fine-tuning
    ├── eval.py                        # Evaluation (QA, Preference, Response)
    └── inference.py                   # Personalized response generation
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended: 2x 32GB Tesla V100 or equivalent)
- 50GB+ disk space for model weights

### Setup

```bash
# Clone the repository
git clone https://github.com/MatthewKKai/MaLP.git
cd MaLP

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"
```

### Model Download

Download the LLaMA-2 model (7B or 13B chat variant):

```bash
# Using Hugging Face CLI
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir mem_llm/llama/Llama-2-7b-chat-hf
```

## Pipeline Stages

The MaLP pipeline consists of six sequential stages. Each stage must be completed before proceeding to the next.

---

### Stage 1: Memory Formation (DPeM)

Forms the Dual-Process enhanced Memory from historical dialogues.

**Process:**
1. **Rehearsal Process (Learning)**: Coordinator C extracts notes from each dialogue iteration
2. **Rehearsal Process (Summarizing)**: Filters and categorizes knowledge as common-sense or user-specific
3. **Executive Process (Memorizing)**: Stores knowledge in STM with type labels
4. **Executive Process (Transit)**: Moves frequently accessed knowledge from STM to LTM (threshold θ)

```bash
cd mem_llm

python memory_formation.py \
    --dialogue_path ../data/dialogues2_cleaned.json \
    --output_dir ./memory_output \
    --transit_threshold 3 \
    --stm_refresh_interval 5 \
    --model gpt-4.1-mini \
    --max_dialogues 0
```

**Or using the script:**
```bash
bash scripts/run_memory_formation.sh
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--dialogue_path` | Path to dialogue JSON file | Required |
| `--output_dir` | Directory to save formed memory | `./memory_output` |
| `--transit_threshold` | Frequency threshold θ for STM→LTM transit | `3` |
| `--stm_refresh_interval` | Iterations between STM transit checks | `5` |
| `--model` | LLM model for coordinator C | `gpt-4.1-mini` |
| `--max_dialogues` | Max dialogues to process (0=all) | `0` |

**Output:** `memory_output/stm.json`, `memory_output/ltm.json`, `memory_output/memory_summary.json`

---

### Stage 2: Data Preparation

Converts dialogue data into training format for both knowledge injection and LoRA fine-tuning.

```bash
cd mem_llm

# Prepare knowledge injection data
python prepare_data.py \
    --dialogue_path ../data/dialogues2_cleaned.json \
    --output_path ./training_data/knowledge_data.json \
    --mode knowledge

# Prepare fine-tuning data (with memory prompts)
python prepare_data.py \
    --dialogue_path ../data/dialogues2_cleaned.json \
    --memory_path ./memory_output \
    --output_path ./training_data/finetune_data.json \
    --mode finetune
```

**Or using the script:**
```bash
bash scripts/run_prepare_data.sh
```

---

### Stage 3: Medical Knowledge Injection (Domain Adaptation)

Injects medical domain knowledge into the base LLM via adapter layers (Section 2.2).

**Architecture:**
- Down-projection: d_model → d_model / 16
- ReLU activation
- Up-projection: d_model / 16 → d_model

**Training objectives:**
- Knowledge loss: L_K = -1/K × Σ log p(m_i)
- Sample loss: L_S = ||V_o - V_k||² (prevents catastrophic forgetting)

```bash
cd mem_llm

python knowledge_injection.py \
    --model_path llama/Llama-2-7b-chat-hf \
    --data_path ./training_data/knowledge_data.json \
    --output_dir ./pretrained_model \
    --epochs 3 \
    --batch_size 20 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --adapter_layers 7,11 \
    --fp16
```

**Or using the script:**
```bash
bash scripts/run_knowledge_injection.sh
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to pre-trained LLaMA model | Required |
| `--data_path` | Path to medical knowledge JSON | Required |
| `--output_dir` | Directory to save adapted model | `./pretrained_model` |
| `--epochs` | Training epochs | `3` |
| `--batch_size` | Batch size | `20` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--weight_decay` | Weight decay | `0.05` |
| `--adapter_layers` | Comma-separated layer indices | `7,11` |
| `--fp16` | Use FP16 mixed precision | Flag |

---

### Stage 4: LoRA Fine-Tuning

Fine-tunes the knowledge-adapted LLM using LoRA on user-specific dialogue data (Section 2.4.2).

**LoRA Configuration:**
- Rank r = 8
- Scaling factor α = 32
- Target modules: q_proj, v_proj
- Dropout: 0.05

```bash
cd mem_llm

# Single GPU
python train.py \
    --data_path ./training_data/finetune_data.json \
    --model_path ./pretrained_model/base_model \
    --output_dir ./finetuned_model \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --fp16

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 train.py \
    --data_path ./training_data/finetune_data.json \
    --model_path ./pretrained_model/base_model \
    --output_dir ./finetuned_model \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --use_ddp \
    --fp16
```

**Or using the script:**
```bash
# Single GPU
bash scripts/run_finetune.sh

# Multi-GPU
NUM_GPUS=2 bash scripts/run_finetune.sh
```

---

### Stage 5: Evaluation

Evaluates the model on three tasks (Section 4.3):

1. **Question Answering (QA)**: Profile QA and Knowledge QA (ROUGE-1, ROUGE-L)
2. **Preference Classification**: Classify user dialogue preference (Accuracy)
3. **Response Generation**: Quality via Win Rate (LLM-as-judge)

```bash
cd mem_llm

# Run all evaluations
python eval.py \
    --dialogue_path ../data/dialogues2_cleaned.json \
    --profiles_path ../dialogue_generation/profiles_4.json \
    --memory_path ./memory_output \
    --output_dir ./eval_results \
    --task all \
    --num_samples 100 \
    --model gpt-4.1-mini

# Run specific task
python eval.py --task qa ...
python eval.py --task preference ...
python eval.py --task response ...
```

**Or using the script:**
```bash
bash scripts/run_eval.sh
```

---

### Stage 6: Inference

Generate personalized responses using the full MaLP pipeline.

```bash
cd mem_llm

# Interactive mode
python inference.py \
    --model_path ./finetuned_model \
    --memory_path ./memory_output \
    --interactive

# Single query
python inference.py \
    --model_path ./finetuned_model \
    --memory_path ./memory_output \
    --query "What should I do about my recurring headaches?" \
    --dialogue_history ../data/dialogues2_cleaned.json
```

**Or using the script:**
```bash
# Interactive
bash scripts/run_inference.sh --interactive

# Single query
bash scripts/run_inference.sh --query "What should I do about my headaches?"
```

---

## Dialogue Generation (Optional)

To generate new personalized dialogue data using self-chat simulation:

```bash
cd dialogue_generation

# Set up API key
export OPENAI_API_KEY="your-api-key-here"

# Generate dialogues
python dialogue_generation.py
```

This uses patient profiles from `profiles_4.json` to simulate doctor-patient conversations with personalized preferences.

---

## Key Components

### DPeM Memory Mechanism

| Memory Type | Refresh | Storage | Supports Lookup | Retriever |
|-------------|---------|---------|-----------------|-----------|
| Working Memory | Each iteration | Limited | No | N/A |
| STM | Certain rounds | Limited | Yes | R_c (Levenshtein) |
| LTM | Never | Unlimited | Yes | R_s (Cosine similarity) |

### Memory Transit Rule

When the frequency of a knowledge item k_i in the flag table reaches the threshold θ, it is transferred from STM to LTM:

```
if frequency(k_i) >= θ:
    LTM[k_i] = STM[k_i]
```

### Retrieval Process

- **STM Retrieval (R_c)**: Closest-match retriever using Levenshtein distance
- **LTM Retrieval (R_s)**: Semantic-match retriever using cosine similarity of sentence embeddings

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | API key for the coordinator LLM | Yes (for Stages 1, 5) |
| `OPENAI_BASE_URL` | Custom API base URL | No |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | No |

### Hardware Requirements

| Stage | Minimum GPU | Recommended |
|-------|-------------|-------------|
| Memory Formation | CPU only | CPU (API-based) |
| Knowledge Injection | 1x 16GB GPU | 1x 32GB GPU |
| LoRA Fine-Tuning | 1x 16GB GPU | 2x 32GB GPU |
| Inference | 1x 16GB GPU | 1x 32GB GPU |

---

## Citation

```bibtex
@inproceedings{zhang-etal-2024-memory,
    title = "Memory-Augmented {LLM} Personalization with Short- and Long-Term Memory Coordination",
    author = "Zhang, Kai and Kang, Yangyang and Zhao, Fubang and Liu, Xiaozhong",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.132",
    pages = "2386--2398",
}
```

## License

This project is for research purposes. Please refer to the original paper for usage guidelines.
