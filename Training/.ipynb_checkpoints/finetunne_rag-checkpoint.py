import os
import torch
import numpy as np
import faiss
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
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

# === Prepare Dataset Class for MCQA Fine-tuning ===
class MCQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        answer_letter = ex['answer'].strip().upper()
        full_answer = f"{answer_letter}. {ex['choices'][ord(answer_letter) - ord('A')].strip()}"
        
        # Build prompt
        prompt = f"Question: {ex['question']}\nOptions: {', '.join(ex['choices'])}\nAnswer:"
        input_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=False)
    
        # Tokenize full answer
        answer_enc = self.tokenizer(" " + full_answer, add_special_tokens=False, return_tensors="pt")  # Prepend space for correct tokenization
    
        input_ids = input_enc["input_ids"].squeeze()
        attention_mask = input_enc["attention_mask"].squeeze()
        answer_ids = answer_enc["input_ids"].squeeze()
    
        # Combine prompt and answer for labels
        total_len = len(input_ids) + len(answer_ids)
        max_len = min(total_len, self.max_length)
    
        # Truncate if needed
        input_ids = torch.cat([input_ids, answer_ids])[:self.max_length]
        attention_mask = torch.ones_like(input_ids)
    
        # Labels: -100 for prompt, actual answer_ids for answer
        labels = torch.full_like(input_ids, -100)
        answer_start = len(input_ids) - len(answer_ids)
        labels[answer_start:answer_start + len(answer_ids)] = answer_ids[:self.max_length - answer_start]
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# === Load Fine-tuning and Validation Datasets ===
print("🔍 Loading datasets...")
train_data = load_dataset(MCQA_DS, 'MMLU', split="train[:1000]")
val_data = load_dataset(MCQA_DS, 'MMLU', split="validation[:500]")

train_dataset = MCQADataset(train_data, tokenizer)
val_dataset = MCQADataset(val_data, tokenizer)

# === Evaluation Metrics ===
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    preds = np.argmax(predictions, axis=-1)

    target_ids = [label[label != -100][0].item() if (label != -100).any() else -1 for label in torch.tensor(labels)]
    pred_ids = [pred[0].item() if isinstance(pred[0], torch.Tensor) else pred[0] for pred in preds]

    acc = accuracy_score(target_ids, pred_ids)
    return {"first_token_accuracy": acc}

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=64,  # Start large; adjust based on memory
    gradient_accumulation_steps=1,   # Increase if you hit OOM and want effective larger batch
    fp16=True,                       # Enables mixed precision (reduce memory usage)
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    push_to_hub=True,
    hub_model_id="igzi/Qwen3-0.6B-answer-first-token",
    device="cpu"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    compute_metrics=compute_metrics,
)

# === Run Training and Push to Hub ===
print("🚀 Starting training...")
trainer.train()
print("📤 Uploading model to Hugging Face Hub...")
trainer.push_to_hub()

print("✅ Done.")
