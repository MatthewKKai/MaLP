#!/bin/bash
#SBATCH --mem 80G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -C A100
#SBATCH -p short

source ../venv_llm/bin/activate

# Example: bash stage_one_pretrain.sh model/test datasets/amazon/train.json test-project 20 1 3 1e-4 2e-4 "" 0

dirname=$1 # the directory to save the pre-trained model
knowledge_path=$2 # path to domain knowledge
project_name=$3 # the name of the project
max_epochs=$4 # the maximum number of epochs
devices=$5 # number of GPUs used
batch_size=$6 # batch size
lr=$7 # learning rate
moa_lr=$8 # learning rate for MoA
realm_record=$9 # path to the REALM record used for information retrieval. Ignore it to disable it
no_old_knowledge=${10} # set it to 1 if you want to disale old domain knowledge

other_params=""
if [[ $no_old_knowledge == 1 ]]; then
    other_params="--no_old_knowledge"
fi
if [[ $realm_record != "" ]]; then
    other_params="${other_params} --realm_record ${realm_record}"
fi

python3 -m pretraining.run_pretraining.py \
    --max_epochs ${max_epochs} \
    --accelerator gpu --strategy ddp \
    --devices ${devices} \
    --batch_size ${batch_size} \
    --layers 7,11 \
    --knowledge_data_path ${knowledge_path} \
    --project_name ${project_name} \
    --run_name ${project_name} \
    --dirpath ${dirname} \
    --lr ${lr} \
    --moe_lr ${moa_lr} \
    --adapter_down_scale 16 \
    ${other_params}
