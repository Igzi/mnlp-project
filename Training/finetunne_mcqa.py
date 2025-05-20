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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Move model to GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# === Load and Split Document Corpus ===
# corpus = load_dataset(DOCUMENTS_DS, split="train")

# splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#     tokenizer=tokenizer,
#     chunk_size=CHUNK_SIZE,
#     chunk_overlap=int(CHUNK_SIZE / 10),
#     add_start_index=True,
#     strip_whitespace=True,
#     separators=MARKDOWN_SEPARATORS,
# )

# print("🔍 Splitting documents...")
# chunks = []
# for doc in corpus:
#     chunks.extend(splitter.split_text(doc["text"]))

# # === Embed Chunks ===
# print("🔍 Embedding document chunks...")
# encoder = SentenceTransformer(ENCODER_NAME)
# embeddings = encoder.encode(
#     chunks, batch_size=64, show_progress_bar=True,
#     convert_to_numpy=True, normalize_embeddings=True
# )

# # === Build FAISS Index ===
# print("🔍 Building FAISS index...")
# dim = embeddings.shape[1]
# index = faiss.IndexFlatIP(dim)
# index.add(embeddings)
# faiss.write_index(index, "faiss_index.index")

LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
class MCQADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        answer_letter = ex['answer'].strip().upper()
        full_answer = f"{answer_letter}. {ex['choices'][ord(answer_letter) - ord('A')].strip()}"
    
        # Build prompt
        topic = "knowledge and skills in advanced master-level STEM courses"
        prompt = (
            f"The following are multiple choice questions (with answers) about {topic}.\n\n"
            f"{ex['question']}\n" +
            "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, ex["choices"])]) +
            "Answer:"
        )
    
        # Tokenize separately
        prompt_enc = self.tokenizer(prompt, add_special_tokens=False)
        answer_enc = self.tokenizer(" " + full_answer, add_special_tokens=False)

        # Truncate if combined length is too long
        total_len = len(prompt_enc["input_ids"]) + len(answer_enc["input_ids"])

        # Combine
        input_ids = prompt_enc["input_ids"] + answer_enc["input_ids"]
        attention_mask = prompt_enc["attention_mask"] + [1] * len(answer_enc["input_ids"])

        # Labels: -100 for prompt part, actual answer tokens
        labels = [-100] * len(prompt_enc["input_ids"]) + answer_enc["input_ids"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# === Load Fine-tuning and Validation Datasets ===
print("🔍 Loading datasets...")
# train_data = concatenate_datasets([
#     load_dataset(MCQA_DS, 'MMLU', split="train").select(range(100)),
#     load_dataset(MCQA_DS, 'ARC-Easy', split="train").select(range(100)),
#     load_dataset(MCQA_DS, 'OpenBookQA', split="train").select(range(100)),
#     load_dataset(MCQA_DS, 'ScienceQA', split="train").select(range(100))
# ])

# # === Validation data ===
# val_data = concatenate_datasets([
#     load_dataset(MCQA_DS, 'MMLU', split="validation").select(range(40)),
#     load_dataset(MCQA_DS, 'ARC-Easy', split="validation").select(range(40)),
#     load_dataset(MCQA_DS, 'OpenBookQA', split="validation").select(range(40)),
#     load_dataset(MCQA_DS, 'ScienceQA', split="validation").select(range(40))
# ])

train_data = concatenate_datasets([
    load_dataset(MCQA_DS, 'MMLU', split="train"),
    load_dataset(MCQA_DS, 'ARC-Easy', split="train"),
    load_dataset(MCQA_DS, 'OpenBookQA', split="train"),
    load_dataset(MCQA_DS, 'ScienceQA', split="train")
])

# === Validation data ===
val_data = concatenate_datasets([
    load_dataset(MCQA_DS, 'MMLU', split="validation"),
    load_dataset(MCQA_DS, 'ARC-Easy', split="validation"),
    load_dataset(MCQA_DS, 'OpenBookQA', split="validation"),
    load_dataset(MCQA_DS, 'ScienceQA', split="validation")
])


train_dataset = MCQADataset(train_data, tokenizer)
val_dataset = MCQADataset(val_data, tokenizer)



training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=1,  # Start large; adjust based on memory
    per_device_eval_batch_size=1,
    fp16=True,                       # Enables mixed precision (reduce memory usage)
    num_train_epochs=3,
    save_total_limit=1,                     # ✅ Keep only latest checkpoint
    save_safetensors=False,                 # ✅ Save in .bin format to reduce size
    logging_dir="./logs",
    push_to_hub=True,
    hub_model_id="igzi/Qwen3-0.6B-answer-first-token",
    eval_steps=1,
    eval_strategy="epoch",
    logging_steps=len(train_dataset),
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding="longest"),
)

# === Run Training and Push to Hub ===
print("🚀 Starting training...")
trainer.train()
print("📤 Uploading model to Hugging Face Hub...")
trainer.push_to_hub()

print("✅ Done.")