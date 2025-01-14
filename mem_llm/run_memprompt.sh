#!/bin/bash
#SBATCH --mem 80G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -C A100
#SBATCH -p short

source ../venv_llm/bin/activate

python3 -m torch.distributed.launch --nproc_per_node=2 run_clm.py \
    --model_name_or_path llama/Llama-2-7b-chat-hf \
    --train_file pretraining_data/train_100k.json \
    --validation_file pretraining_data/validation_10k.json \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /results \
    --train_adapter \
    --adapter_config lora \
    --overwrite_output_dir
