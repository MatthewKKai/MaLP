# from config import parser
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
local_rank = int(os.environ["LOCAL_RANK"])
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import torch.distributed as dist
from torch.utils.data import Dataset, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from model.lora_llama import lora_llama
import transformers
import torch
import json

# initilization
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)

data_path = r"dialogues.json"
llama_path = r"llama/Llama-2-7b-chat-hf"

print(torch.cuda.is_available())

class Dialogues(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.labels = []
        self.input_ids = []
        self.attn_masks = []
        for txt in txt_list:
            encoding_dict_input = tokenizer(txt["input"], truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encoding_dict_input["input_ids"]))
            self.attn_masks.append(torch.tensor(encoding_dict_input["attention_mask"]))
            encoding_dict_output = tokenizer(txt["output"], truncation=True, max_length=max_length, padding="max_length")
            self.labels.append(torch.tensor(encoding_dict_output["input_ids"]))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

# obtain lora_llama, tokenizer

# print("------------------{0}-------------------{1}".format(torch.cuda.device_count(), torch.cuda.is_available()))
llama = LlamaForCausalLM.from_pretrained(llama_path)
lora_llama = lora_llama(llama).get_lora_llama()
lora_llama.print_trainable_parameters()
tokenizer = LlamaTokenizer.from_pretrained(llama_path)
tokenizer.pad_token = tokenizer.eos_token
# model = lora_llama.to(device)
# model.config.use_cache = False
ddp_model = DDP(lora_llama.to(device), device_ids=[local_rank])
# torch.cuda.empty_cache()

# get data ready
# data = []
# for dir in os.listdir(r"user_specific/"):
#     with open(os.path.join(r"user_specific", dir), encoding="utf-8") as f:
#         data_tmp =json.load(f)
#         data+=data_tmp

# dialogues = []
# for dialogue in data:
#     tmp = str(dialogue).strip("[").strip("]")
#     dialogues.append(tmp)

with open(r"pretraining_data/combined_110k.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

dataset = Dialogues(dialogues, tokenizer, 768)
train_dataset, val_dataset = random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
input_ids_test, attn_masks_test, label = dataset.__getitem__(0)
# print(input_ids_test.shape)
# print(attn_masks_test.shape)
# print(dataset.__len__())

input_test_ = tokenizer("How to treat fever?", return_tensors="pt").input_ids.cuda()
sample_outputs_before = ddp_model.module.generate(input_ids=input_test_,
    do_sample=True,
    top_k=50,
    max_length=1500,
    top_p=0.95,
    temperature=1.0)
generated_txt_ = tokenizer.batch_decode(sample_outputs_before, skip_special_tokens=True)
# print(generated_txt_)

# training settings
training_args = TrainingArguments(
    save_steps = 500,
    warmup_steps = 10,
    logging_steps = 100,
    weight_decay = 0.05,
    num_train_epochs = 1,
    logging_dir = "logs/",
    output_dir = "results/",
    # local_rank = int(os.environ.get("LOCAL_RANK", -1)),
    per_device_eval_batch_size = 1,
    per_device_train_batch_size = 1,
    report_to=None
)
trainer = Trainer(
    model=ddp_model,
    args = training_args,
    train_dataset = train_dataset,
    data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]), 'labels': torch.stack([f[2] for f in data])}
)
torch.cuda.empty_cache()
trainer.train()
# print(trainer.evaluate())

# test = dialogues[100]["input"]
test = r"How to treat fever?"
print(test)
input_test = tokenizer("{0}".format(test), return_tensors="pt").input_ids.cuda()
sample_outputs = ddp_model.module.generate(input_ids=input_test,
    do_sample=True, 
    top_k=50,
    max_length=1500,
    top_p=0.95,
    temperature=1.0)
generated_txt = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
print(generated_txt_)
print(generated_txt)