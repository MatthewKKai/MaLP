# MaLP
Implementation Code for "LLM-based Medical Assistant Personalization with Short- and Long-Term Memory Coordination"


## Dataset

We first derive patient's profile from public medical corpus (https://github.com/UCSD-AI4H/Medical-Dialogue-System) and then endow the patient's profile to a powerful chat model. Assistant role (e.g., doctor) will be simulated independently using the same chat model and thus we could collect the historical dialogues via \textit{self-chat} between these two roles.
![data_collection](https://github.com/user-attachments/assets/8b520849-44ee-4c47-b427-2c234254133e)


### Data Sturcture
Profile - Contains personal information such as preference etc.

Dialogue - Contains multi-round dialogue that generated via self-chat.

## Training

### Pre-training 
```
sbatch run_pretraining.sh
```

### Fine-tuning
```
torchrun python mem_llm/train.py
```
