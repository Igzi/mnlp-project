import os
import torch
import numpy as np
import faiss
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

# === Constants ===
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
ENCODER_NAME = "igzi/MNLP_M2_document_encoder"
DOCUMENTS_DS = "igzi/pile-stem-corpus-small-semantic"
MCQA_DS = "igzi/MNLP_M2_mcqa_dataset"
CHUNK_SIZE = 512
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
]

# === Load Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True,
                padding_side="left",
                truncation_side="left",)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

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
            "options": [f" {letter}." for letter in LETTER_INDICES[:len(ex["choices"])]],
            "correct_idx": correct_index,
        }

class MCQATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        prompts = inputs["prompt"]
        completions = inputs["options"]
        correct_idxs = inputs["correct_idx"]

        batch_size = len(prompts)
        device = model.device
        tokenizer = self.tokenizer
        option_logits = []

        for i in range(batch_size):
            prompt = prompts[i]
            options = completions[i]

            logits = []
            for opt in options:
                # Encode prompt + option
                enc = tokenizer(prompt + opt, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    labels = enc["input_ids"].clone()
                    output = model(**enc, labels=labels)
                    nll = output.loss * labels.size(1)  # Total neg log likelihood
                logits.append(-nll)

            option_logits.append(torch.stack(logits))

        logits = torch.stack(option_logits)  # shape: (batch, num_options)
        targets = torch.tensor(correct_idxs, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, targets)

        return (loss, logits) if return_outputs else loss


# === Load Fine-tuning and Validation Datasets ===
print("🔍 Loading datasets...")
train_data = concatenate_datasets([
    load_dataset(MCQA_DS, 'MMLU', split="train").select(range(100)),
    # load_dataset(MCQA_DS, 'ARC-Easy', split="train").select(range(100)),
    # load_dataset(MCQA_DS, 'OpenBookQA', split="train").select(range(100)),
    # load_dataset(MCQA_DS, 'ScienceQA', split="train").select(range(100))
])

# === Validation data ===
val_data = concatenate_datasets([
    load_dataset(MCQA_DS, 'MMLU', split="validation").select(range(40)),
    # load_dataset(MCQA_DS, 'ARC-Easy', split="validation").select(range(40)),
    # load_dataset(MCQA_DS, 'OpenBookQA', split="validation").select(range(40)),
    # load_dataset(MCQA_DS, 'ScienceQA', split="validation").select(range(40))
])

# train_data = concatenate_datasets([
#     load_dataset(MCQA_DS, 'MMLU', split="train"),
#     # load_dataset(MCQA_DS, 'ARC-Easy', split="train"),
#     # load_dataset(MCQA_DS, 'OpenBookQA', split="train"),
#     # load_dataset(MCQA_DS, 'ScienceQA', split="train")
# ])

# # === Validation data ===
# val_data = concatenate_datasets([
#     load_dataset(MCQA_DS, 'MMLU', split="validation"),
#     # load_dataset(MCQA_DS, 'ARC-Easy', split="validation"),
#     # load_dataset(MCQA_DS, 'OpenBookQA', split="validation"),
#     # load_dataset(MCQA_DS, 'ScienceQA', split="validation")
# ])


train_dataset = MCQADatasetClassification(train_data, tokenizer)
val_dataset = MCQADatasetClassification(val_data, tokenizer)

sample = train_dataset[0]  # Should return dict with "prompt", "options", "correct_idx"
print(sample) 

def collatefn(batch):
    print("Here: ", batch)
    return {
        "prompt": [item["prompt"] for item in batch],
        "options": [item["options"] for item in batch],
        "correct_idx": [item["correct_idx"] for item in batch],
    }

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    gradient_accumulation_steps = 8,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    bf16=True,                       # Enables mixed precision (reduce memory usage)
    num_train_epochs=2,
    save_total_limit=3,                     # ✅ Keep only latest checkpoint
    save_safetensors=False,                 # ✅ Save in .bin format to reduce size
    logging_dir="./logs",
    push_to_hub=True,
    hub_model_id="igzi/Qwen3-0.6B-answer-first-token",
    eval_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    max_grad_norm=0.5,
    logging_steps=len(train_dataset),
)

# === Trainer ===
trainer = MCQATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collatefn,
)

# === Run Training and Push to Hub ===
print("🚀 Starting training...")
trainer.train()
print("📤 Uploading model to Hugging Face Hub...")
trainer.push_to_hub()

print("✅ Done.")