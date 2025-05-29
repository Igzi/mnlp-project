import os
import torch
import numpy as np
import math
import faiss
import pickle
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
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
from langchain_community.vectorstores import FAISS
from torch.utils.data import DataLoader
from torch.nn import functional as F

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
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16)
# Move model to GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

model_kwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", # Dynamically check cuda
}
encode_kwargs = {
    "normalize_embeddings": True
}
embedding_model = HuggingFaceEmbeddings(
            model_name=ENCODER_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
# === Load FAISS Index and Supporting Files ===
print("📥 Loading FAISS index and metadata...")
index = faiss.read_index("faiss_index.index")

with open("docstore.pkl", "rb") as f:
    docstore = pickle.load(f)

with open("index_to_docstore_id.pkl", "rb") as f:
    index_to_docstore_id = pickle.load(f)
    vector_db = FAISS(
            embedding_function=embedding_model.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            distance_strategy=DistanceStrategy.COSINE,
            normalize_L2=True
        )

print(vector_db.index.ntotal)

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
            f"{ex['question']}\n" +
            "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, ex["choices"])]) +
            "Answer:"
        )

        D = vector_db.similarity_search(query=prompt, k=3)

        retrieved_docs_text = [doc.page_content for doc in D]
        context = "\nRelavent Documents:\n"
        context += "\n\n".join([
            f"Document {str(i)}:::\n" + doc
            for i, doc in enumerate(retrieved_docs_text)
        ])

        prompt = (
            "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n"+
            context +
            prompt
        )

        return {
            "prompt": prompt,
            "options": [f" {letter}" for letter in LETTER_INDICES[:len(ex["choices"])]],
            "correct_idx": correct_index,
            "dataset": ex["dataset"]
        }


class MCQATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        For each prompt we run the model once, grab the final (next‐token)
        logits, index into the letter‐token IDs, and compute a CE loss.
        """
        device = model.device
        prompts      = inputs["prompt"]       # List[str]
        correct_idxs = inputs["correct_idx"]  # List[int]
        all_options  = inputs["options"]      # List[List[str]]

        batch_logits = []
        losses = []

        # Pre‐tokenize all option‐letters to single token IDs
        option_token_ids = [
            [ tokenizer(opt, add_special_tokens=False).input_ids[0]
              for opt in opts ]
            for opts in all_options
        ]

        for prompt, opt_ids, target in zip(prompts, option_token_ids, correct_idxs):
            # 1) encode prompt
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=2048
            ).to(device)

            # 2) single forward pass
            outputs = model(**enc)
            # outputs.logits: [1, seq_len, vocab_size]
            last_logits = outputs.logits[:, -1, :]            # [1, V]

            # 3) pick out the logits for our option tokens → [1, num_opts]
            opt_logits = last_logits[:, opt_ids]               # [1, num_opts]
            batch_logits.append(opt_logits.squeeze(0))         # [num_opts]

            # 4) CE against the correct index
            tgt = torch.tensor([target], device=device)
            losses.append(F.cross_entropy(opt_logits, tgt))

            del enc, outputs, last_logits, opt_logits, tgt
            torch.cuda.empty_cache()

        loss = torch.stack(losses).mean()
        logits = torch.stack(batch_logits)                   # [batch, num_opts]

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

        # track per‐dataset stats
        correct_by_ds = {}
        total_by_ds   = {}

        # overall stats
        overall_correct = 0
        overall_total   = 0

        with torch.inference_mode(), torch.cuda.amp.autocast():
            for batch in dataloader:
                prompts      = batch["prompt"]
                options      = batch["options"]
                correct_idxs = batch["correct_idx"]
                datasets     = batch["dataset"]

                for i in range(len(prompts)):
                    ds_name = datasets[i]
                    prompt  = prompts[i]
                    opts    = options[i]
                    target  = correct_idxs[i]

                    # ensure counters exist
                    if ds_name not in correct_by_ds:
                        correct_by_ds[ds_name] = 0
                        total_by_ds[ds_name]   = 0

                    # score each option by negative NLL
                    scores = []
                    for opt in opts:
                        enc = tokenizer(
                            prompt + opt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=2048
                        ).to(device)
                        labels = enc["input_ids"].clone()
                        out = model(**enc, labels=labels)
                        nll = out.loss * labels.size(1)
                        scores.append(-nll.item())
                        del enc, labels, out
                        torch.cuda.empty_cache()

                    pred = int(torch.argmax(torch.tensor(scores)))

                    # update stats
                    is_correct = (pred == target)
                    correct_by_ds[ds_name] += int(is_correct)
                    total_by_ds[ds_name]   += 1
                    overall_correct += int(is_correct)
                    overall_total   += 1

        # compute accuracies
        acc_by_ds = {
            ds: correct_by_ds[ds] / total_by_ds[ds]
            for ds in correct_by_ds
        }
        overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

        # print results
        print(f"→ Overall accuracy: {overall_acc:.4f} " 
              f"({overall_correct}/{overall_total})")
        for ds, acc in acc_by_ds.items():
            c, t = correct_by_ds[ds], total_by_ds[ds]
            print(f"→ {ds} accuracy: {acc:.4f} ({c}/{t})")

        # return as metrics dict
        metrics = {"accuracy": overall_acc}
        metrics.update({f"accuracy_{ds}": acc for ds, acc in acc_by_ds.items()})
        return metrics



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
    load_dataset(MCQA_DS, 'ARC-Easy', split="train"),
    load_dataset(MCQA_DS, 'ARC-Challenge', split="train"),
    # load_dataset(MCQA_DS, 'OpenBookQA', split="train"),
    # load_dataset(MCQA_DS, 'ScienceQA', split="train")
])

# === Validation data ===
val_data = concatenate_datasets([
    load_dataset(MCQA_DS, 'MMLU', split="validation"),
    load_dataset(MCQA_DS, 'ARC-Easy', split="validation"),
    load_dataset(MCQA_DS, 'ARC-Challenge', split="validation"),
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
        "dataset": [item["dataset"] for item in batch],
    }

steps_per_epoch = len(train_dataset) // 64
half_epoch_steps = steps_per_epoch // 2

training_args = TrainingArguments(
    output_dir="./finetuned_full_dataset_rag_new",
    gradient_accumulation_steps = 64,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-5,
    bf16=True,                       # Enables mixed precision (reduce memory usage)
    num_train_epochs=3,
    save_safetensors=False,                 # ✅ Save in .bin format to reduce size
    logging_dir="./logs",
    push_to_hub=True,
    hub_model_id="igzi/Qwen3-0.6B-full-dataset-mcqa",
    eval_strategy="steps",                  # ⬅️ switched from 'epoch'
    save_strategy="steps",                  # ⬅️ switched from 'epoch'
    eval_steps=half_epoch_steps,            # ⬅️ half epoch
    save_steps=half_epoch_steps,            # ⬅️ half epoch
    max_grad_norm=0.5,
    logging_steps=len(train_dataset),
    warmup_ratio = 0.05,
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
# print("📤 Uploading model to Hugging Face Hub...")
# trainer.push_to_hub()

# print("✅ Done.")