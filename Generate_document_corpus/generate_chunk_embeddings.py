import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load model and move to GPU
device = torch.device("cuda")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
model = model.to(device)

# Load dataset
filtered_data = load_dataset("igzi/pile-stem-corpus-filtered", split="train")
subset = filtered_data.shuffle(seed=42).select(range(500_000))
texts = [example for example in subset]

# Embedding function with tqdm
def embed_tokenized_chunks(model, tokenized_texts, max_batch_size=1024):
    embeddings = []
    batch_size = max_batch_size
    idx = 0

    pbar = tqdm(total=len(tokenized_texts), desc="Embedding")

    while idx < len(tokenized_texts):
        try:
            end_idx = min(idx + batch_size, len(tokenized_texts))
            batch = tokenized_texts[idx:end_idx]
            input_ids = [item["input_ids"] for item in batch]
            attention_mask = [item.get("attention_mask", [1] * len(item["input_ids"])) for item in batch]

            batch_tokenized = [
                {"input_ids": ids, "attention_mask": mask}
                for ids, mask in zip(input_ids, attention_mask)
            ]

            emb = model.encode(
                batch_tokenized,
                normalize_embeddings=True,
                show_progress_bar=False,
                is_pretokenized=True,
            )
            embeddings.append(emb)
            pbar.update(end_idx - idx)
            idx = end_idx
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                print(f"[WARN] CUDA OOM: Reducing batch size to {batch_size}")
            else:
                raise e

    pbar.close()
    return np.vstack(embeddings)

# Run embedding with progress
embeddings = embed_tokenized_chunks(model, texts)
embedding_list = embeddings.tolist()

# Add to dataset
subset = subset.add_column("embedding", embedding_list)

# Save locally
subset.save_to_disk("pile-stem-corpus-filtered-embedded")

# Push to Hub
subset.push_to_hub("igzi/pile-stem-corpus-filtered-embedded")
