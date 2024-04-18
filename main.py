import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer
from peft import LoraConfig
import transformers
from trl import SFTTrainer
from dotenv import load_dotenv
from datasets import load_dataset
import wandb

load_dotenv()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "google/gemma-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, device=device, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])

TEMPLATE = "Given the following cryptic clue for a british cryptic crossword, what is the answer?\nClue:\n{clue}\n\nAnswer:\n{answer}"

class TrainDataset(Dataset):
  def __init__(self):
    self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-7b")
    self.tokenizer.pad_token_id = 0 
    self.tokenizer.padding_side = "right"

    self.data_df = pd.read_csv("cryptics.csv")
    self.data_df = self.data_df.drop(columns=["index", "definition" ,"location", "date" ,"source" ,"source_address", "source_slug"])

    self.ds_prompts = self.data_df.apply(self.prompt,  axis=1)
    self.ds = self.ds_prompts.apply(self.tokenize)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    return self.ds.iloc[idx]

  def prompt(self, elm):
    prompt = TEMPLATE.format(clue=elm["clue"], answer=elm["answer"])
    return {"prompt": prompt}

  def tokenize(self, elm):
    res = self.tokenizer(elm["prompt"])
    res["input_ids"].append(self.tokenizer.eos_token_id)
    res["attention_mask"].append(1)
    res["labels"] = res["input_ids"].copy()
    return res

  def max_seq_len(self):
    return max([len(elm["input_ids"]) for elm in self.ds])

def example(model, tokenizer):
    text = "Quote: Imagination is more"
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
torch.cuda.empty_cache()

ds = TrainDataset()
collator = transformers.DataCollatorForSeq2Seq(ds.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

wandb.init(
    # set the wandb project where this run will be logged
    project="cryptic-gemma",
    name="cryptic-gemma",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-4,
    "architecture": "GEMMA7B",
    "dataset": "CRYPTICSV_DATASET",
    "epochs": 1,
    "batch_size": 1,
    "optim": "paged_adamw_8bit",
    "device": device.type,
    }
)


def formatting_func(example):
    text = f"""Given the following cryptic clue for a British crytpic crossword, provide the corresponding answer.
    Clue:
    {example["clue"]}
    Answer:
    {example["answer"]}
    """
    print(text)
    return [text]


trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=transformers.TrainingArguments(
        num_train_epochs = 1,
        per_device_train_batch_size=1,
        # gradient_accumulation_steps=16,
        # warmup_steps=2,
        # max_steps=2600,
        learning_rate=2e-4,
        fp16=True,
        seed = 12,
        logging_steps=1,
        evaluation_strategy="no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=200,
        output_dir="./outputs",
        save_total_limit=3,
        optim="paged_adamw_8bit",
        report_to="wandb",
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)

model.config.use_cache = False 
trainer.train()
model.save_pretrained("./weights")