import os
import torch
import numpy as np
import math
import faiss
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
corpus = load_dataset(DOCUMENTS_DS, split="train")

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=int(CHUNK_SIZE / 10),
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

print("🔍 Splitting documents...")
chunks = []
for doc in corpus:
    chunks.extend(splitter.split_text(doc["text"]))

# === Embed Chunks ===
print("🔍 Embedding document chunks...")
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
embeddings = embedding_model.embed_documents(chunks)

# === Build FAISS Index ===
print("🔍 Building FAISS index...")
N = len(chunks)

if N == 0:
    raise ValueError("Cannot build an index with 0 vectors.")
elif N < 8:
    # For very small N, IVF might be overkill or unstable.
    # A small fixed nlist or even falling back to Flat might be better.
    # Let's use a minimum of 4 lists if N is sufficient, otherwise just 1.
    nlist = min(4, N)
    print(f"Very small N ({N}). Using nlist={nlist}. Consider using IndexFlatL2 instead.")
else:
    sqrt_n = math.sqrt(N)
    log2_sqrt_n = math.log2(sqrt_n)

    pow2_low = 2**math.floor(log2_sqrt_n)
    pow2_high = 2**math.ceil(log2_sqrt_n)

    if sqrt_n - pow2_low < pow2_high - sqrt_n: # Check distance, prefer lower if equidistant
        closest_pow2 = pow2_low
    else:
        closest_pow2 = pow2_high

    nlist = 4 * closest_pow2
    nlist = max(4, nlist)
    nlist = min(nlist, N)

nlist = int(nlist)
print(f"Using N={N}, calculated nlist = {nlist} (based on 2 * closest power of 2 to sqrt(N))")

dim = len(embeddings[0])
pq_m = min(32, dim)
index_key = f"IVF{nlist},PQ{pq_m}"
try:
    index = faiss.index_factory(dim, index_key)
except Exception as e:
     print(f"Failed to create index factory for {index_key}. Dim={dim}. Error: {e}")
     # Fallback to FlatL2 if IVF-PQ fails (e.g., dim < pq_m or too few vectors for nlist)
     print("Falling back to IndexFlatL2 due to index_factory error.")
     index = faiss.IndexFlatL2(dim)
     index_key = "IndexFlatL2"

# Check if training is required (IVF needs training, Flat does not)
if hasattr(index, 'is_trained') and not index.is_trained:
    print(f"Training FAISS index ({index_key})...")
    embeddings_np = np.asarray(embeddings, dtype="float32")
    if embeddings_np.shape[0] < nlist and index_key.startswith("IVF"):
         print(f"Insufficient vectors ({embeddings_np.shape[0]}) to train {nlist} clusters. Reducing nlist.")
         # Adjust nlist or fallback - simple fallback shown here
         index = faiss.IndexFlatL2(dim)
         index_key = "IndexFlatL2 (fallback)"
         print(f"Switched to {index_key} due to insufficient training data.")
    elif index_key.startswith("IVF"): # Only train if IVF and enough data
        res = faiss.StandardGpuResources()
        device_id = 0  # or whichever GPU you want
        index = faiss.index_cpu_to_gpu(res, device_id, index)
        index.train(embeddings_np)

documents = [Document(page_content=chunk) for chunk in chunks]

# Generate unique IDs for each chunk (e.g., simple stringified index)
doc_ids = [str(i) for i in range(len(documents))]

# Map FAISS index to docstore IDs
index_to_docstore_id = {i: doc_ids[i] for i in range(len(doc_ids))}
docstore = InMemoryDocstore(dict(zip(doc_ids, documents)))

print("💾 Saving FAISS index and docstore...")

# Save FAISS index to disk
faiss.write_index(faiss.index_gpu_to_cpu(index), "faiss_index.index")

# Save docstore and index mapping
import pickle
with open("docstore.pkl", "wb") as f:
    pickle.dump(docstore, f)

with open("index_to_docstore_id.pkl", "wb") as f:
    pickle.dump(index_to_docstore_id, f)

print("✅ Vector database saved to disk.")