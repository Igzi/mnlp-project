import os
import torch
import numpy as np
import faiss
from torch.nn import functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorWithPadding
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# === Constants ===
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MCQA_DS = "igzi/MNLP_M2_mcqa_dataset"
CHUNK_SIZE = 512
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
]

# === Load Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True,
                padding_side="left",
                truncation_side="left",)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision="main", trust_remote_code=True)

LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

class MCQADatasetClassification(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        answer_letter = ex["answer"].strip().upper()
        correct_index = ord(answer_letter) - ord("A")

        prompt = (
            "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n"
            f"{ex['question']}\n" +
            "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, ex["choices"])]) +
            "Answer:"
        )

        return {
            "prompt": prompt,
            "options": [f" {letter}" for letter in LETTER_INDICES[:len(ex["choices"])]],
            "correct_idx": correct_index,
        }

class MCQATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        prompts = inputs["prompt"]
        completions = inputs["options"]
        correct_idxs = inputs["correct_idx"]

        batch_size = len(prompts)
        device = model.device
        option_logits = []

        with torch.cuda.amp.autocast():
            for i in range(batch_size):
                prompt = prompts[i]
                options = completions[i]
    
                logits = []
                for opt in options:
                    # Encode prompt + option
                    enc = tokenizer(
                        prompt + opt, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=2048
                    ).to(device)
                    
                    with torch.no_grad():  # No need for gradients on labels
                        labels = enc["input_ids"].clone()
                    output = model(**enc, labels=labels)
                    nll = output.loss * labels.size(1)  # Total neg log likelihood
                    logits.append(-nll)
                    del enc, labels, output
                    torch.cuda.empty_cache()
    
                option_logits.append(torch.stack(logits))

        logits = torch.stack(option_logits)  # shape: (batch, num_options)
        targets = torch.tensor(correct_idxs, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, targets)

        return (loss, logits) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        dataloader_params = {
            "batch_size": 1,
            "collate_fn": self.data_collator
        }
        return DataLoader(self.train_dataset, **dataloader_params)

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        dataloader_params = {
            "batch_size": 1,
            "collate_fn": self.data_collator
        }
        return DataLoader(eval_dataset, **dataloader_params)

    def evaluate(self, ignore_keys=None):
        model = self.model
        model.eval()
        dataloader = self.get_eval_dataloader(self.eval_dataset)
        device = model.device
    
        correct = 0
        total = 0
    
        # Use inference mode and mixed precision for evaluation
        with torch.inference_mode(), torch.cuda.amp.autocast():
            for batch in dataloader:
                prompts = batch["prompt"]
                options = batch["options"]
                correct_idxs = batch["correct_idx"]
    
                for i in range(len(prompts)):
                    prompt = prompts[i]
                    opts = options[i]
                    target = correct_idxs[i]
                    scores = []
    
                    for opt in opts:
                        # Encode with truncation to prevent OOM
                        enc = tokenizer(
                            prompt + opt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=2048
                        ).to(device)
                        
                        labels = enc["input_ids"].clone()
                        
                        # Forward pass - no need for output object storage
                        with torch.no_grad():
                            output = model(**enc, labels=labels)
                            nll = output.loss * labels.size(1)
                        
                        scores.append(-nll.item())
                        
                        # Explicit cleanup
                        del enc, labels, output
                        torch.cuda.empty_cache()
    
                    pred = int(torch.argmax(torch.tensor(scores)))
                    correct += (pred == target)
                    total += 1
    
        accuracy = correct / total if total > 0 else 0.0
        print("Evaluation accuracy: ", accuracy)
        return {"accuracy": accuracy}


# === Load Fine-tuning and Validation Datasets ===
print("🔍 Loading datasets...")
# train_data = concatenate_datasets([
#     load_dataset(MCQA_DS, 'MMLU', split="train").select(range(1000)),
#     # load_dataset(MCQA_DS, 'ARC-Easy', split="train").select(range(100)),
#     # load_dataset(MCQA_DS, 'OpenBookQA', split="train").select(range(100)),
#     # load_dataset(MCQA_DS, 'ScienceQA', split="train").select(range(100))
# ])

# # === Validation data ===
# val_data = concatenate_datasets([
#     load_dataset(MCQA_DS, 'MMLU', split="validation").select(range(100)),
#     # load_dataset(MCQA_DS, 'ARC-Easy', split="validation").select(range(40)),
#     # load_dataset(MCQA_DS, 'OpenBookQA', split="validation").select(range(40)),
#     # load_dataset(MCQA_DS, 'ScienceQA', split="validation").select(range(40))
# ])

train_data = concatenate_datasets([
    load_dataset(MCQA_DS, 'MMLU', split="train"),
    # load_dataset(MCQA_DS, 'ARC-Easy', split="train"),
    # load_dataset(MCQA_DS, 'OpenBookQA', split="train"),
    # load_dataset(MCQA_DS, 'ScienceQA', split="train")
])

# === Validation data ===
val_data = concatenate_datasets([
    load_dataset(MCQA_DS, 'MMLU', split="validation"),
    # load_dataset(MCQA_DS, 'ARC-Easy', split="validation"),
    # load_dataset(MCQA_DS, 'OpenBookQA', split="validation"),
    # load_dataset(MCQA_DS, 'ScienceQA', split="validation")
])


train_dataset = MCQADatasetClassification(train_data, tokenizer)
val_dataset = MCQADatasetClassification(val_data, tokenizer)
train_data = train_data.shuffle(seed=42)
val_data = val_data.shuffle(seed=42)

def collatefn(batch):
    return {
        "prompt": [item["prompt"] for item in batch],
        "options": [item["options"] for item in batch],
        "correct_idx": [item["correct_idx"] for item in batch],
    }

steps_per_epoch = len(train_dataset) // 64
half_epoch_steps = steps_per_epoch // 2

training_args = TrainingArguments(
    output_dir="./finetuned_model2",
    gradient_accumulation_steps = 64,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-5,
    bf16=True,                       # Enables mixed precision (reduce memory usage)
    num_train_epochs=3,
    save_safetensors=False,                 # ✅ Save in .bin format to reduce size
    logging_dir="./logs",
    push_to_hub=True,
    hub_model_id="igzi/Qwen3-0.6B-answer-first-token2",
    eval_strategy="steps",                  # ⬅️ switched from 'epoch'
    save_strategy="steps",                  # ⬅️ switched from 'epoch'
    eval_steps=half_epoch_steps,            # ⬅️ half epoch
    save_steps=half_epoch_steps,            # ⬅️ half epoch
    max_grad_norm=0.5,
    logging_steps=len(train_dataset),
    warmup_ratio = 0.03,
    gradient_checkpointing=True,
)


# === Trainer ===
trainer = MCQATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collatefn,
)

trainer.evaluate()
# === Run Training and Push to Hub ===
print("🚀 Starting training...")
trainer.train()
print("📤 Uploading model to Hugging Face Hub...")
trainer.push_to_hub()

# print("✅ Done.")