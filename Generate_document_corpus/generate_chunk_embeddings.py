import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load model and move to GPU
device = torch.device("cuda")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
model = model.to(device)

print("Loading data...")
# Load dataset
filtered_data = load_dataset("igzi/pile-stem-corpus-filtered", split="train").select(range(1500000, 2000000))
print("Data loaded")
filtered_data = filtered_data.remove_columns(["input_ids", "attention_mask"])
texts = [text for text in filtered_data]

def embed_text_chunks(model, texts, max_batch_size=1024):
    embeddings = []
    batch_size = max_batch_size
    idx = 0

    pbar = tqdm(total=len(texts), desc="Embedding")

    while idx < len(texts):
        try:
            end_idx = min(idx + batch_size, len(texts))
            batch = texts[idx:end_idx]

            emb = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                is_pretokenized=False  # default; can omit
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
print("Embedding documents...")
embeddings = embed_text_chunks(model, texts)
embedding_list = embeddings.tolist()

# Add to dataset
subset = filtered_data.add_column("embedding", embedding_list)

# Save locally
subset.save_to_disk("pile-stem-corpus-filtered-embedded4")

# Push to Hub
#subset.push_to_hub("igzi/pile-stem-corpus-filtered-embedded")
