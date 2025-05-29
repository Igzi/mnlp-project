import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import pickle

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)
from datasets import load_dataset, concatenate_datasets

# === Constants ===
MODEL_NAME = "igzi/Qwen3-0.6B-finetuned-rag-mmlu-arc"
ENCODER_NAME = "igzi/MNLP_M2_document_encoder"
MCQA_DS = "igzi/MNLP_M2_mcqa_dataset"
CHUNK_SIZE = 512
K = 5                # number of docs to retrieve
MARGIN = 0.1         # hinge loss margin
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 1e-5
LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# === Setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and freeze LLM ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
llm.eval()
for p in llm.parameters():
    p.requires_grad = False

# === Load encoder ===
tokenizer_enc = AutoTokenizer.from_pretrained(ENCODER_NAME, trust_remote_code=True)
encoder = AutoModel.from_pretrained(ENCODER_NAME).to(device)
encoder.train()
encoder_optimizer = optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)

# === Load FAISS index and docstore ===
print("🗄 Loading FAISS index...")
index = faiss.read_index("faiss_index.index")
with open("docstore.pkl", "rb") as f:
    docstore = pickle.load(f)
with open("index_to_docstore_id.pkl", "rb") as f:
    index_to_docstore_id = pickle.load(f)

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(
    model_name=ENCODER_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
).embed_query
vector_db = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    distance_strategy=DistanceStrategy.COSINE,
    normalize_L2=True
)

# === Load MCQA datasets ===
print("📥 Loading MCQA datasets...")
train_raw = concatenate_datasets([
    load_dataset(MCQA_DS, 'MMLU', split="train"),
    load_dataset(MCQA_DS, 'ARC-Easy', split="train"),
    load_dataset(MCQA_DS, 'ARC-Challenge', split="train")
])
val_raw = concatenate_datasets([
    load_dataset(MCQA_DS, 'MMLU', split="validation"),
    load_dataset(MCQA_DS, 'ARC-Easy', split="validation"),
    load_dataset(MCQA_DS, 'ARC-Challenge', split="validation")
])

# === Simple IterableDataset for questions ===
class QuestionDataset(torch.utils.data.IterableDataset):
    def __init__(self, ds): self.ds = ds
    def __iter__(self):
        for ex in self.ds:
            yield ex

train_loader = DataLoader(QuestionDataset(train_raw), batch_size=None)
val_loader   = DataLoader(QuestionDataset(val_raw),   batch_size=None)

# === Helper: compute CE losses ===
import torch
import torch.nn.functional as F

def get_ce_losses_for_docs(question, correct_idx, options, docs):
    """
    question:   dict with key 'question' → str
    correct_idx: int, index into options for the right answer
    options:    List[str], all the answer choices
    docs:       List[str], retrieved documents to condition on
    returns:    List[float], CE loss per doc
    """
    device = next(llm.parameters()).device

    # 1) pre‐tokenize each option to its single-token ID
    #    (assumes each option like "A", "B", or a single word maps to one token)
    option_token_ids = [
        tokenizer(opt, add_special_tokens=False).input_ids[0]
        for opt in options
    ]

    ce_losses = []
    for doc in docs:
        # 2) build prompt
        prompt = (
            "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n"+
            f"Relevant Documents:\n{doc}\n\n"
            f"{question}\n"
        )
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=2048
        ).to(device)

        # 3) forward pass to get logits
        outputs = llm(**enc)
        # [1, seq_len, vocab_size] → pick last token
        last_logits = outputs.logits[:, -1, :]      # shape [1, V]

        # 4) select logits for our option tokens → [1, num_options]
        opt_logits = last_logits[:, option_token_ids]  # [1, N_opts]

        # 5) CE loss against the correct index
        tgt = torch.tensor([correct_idx], device=device)
        loss = F.cross_entropy(opt_logits, tgt)

        ce_losses.append(loss.item())

        # cleanup
        del enc, outputs, last_logits, opt_logits, tgt
        torch.cuda.empty_cache()

    return ce_losses


# === Evaluation function ===
def evaluate():
    llm.eval()
    correct = 0
    total = 0
    for ex in tqdm(val_loader, desc="Evaluating"): 
        question, choices, answer = ex['question'], ex['choices'], ex['answer'].strip().upper()
        # retrieve with current encoder
        docs = [d.page_content for d in vector_db.similarity_search(query=question, k=2)]
        # build prompt base
        context = "\nRelevant Documents:\n" + "\n\n".join([f"Doc{i}:::\n{d}" for i,d in enumerate(docs)])
        base = (
            "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n" + context + question + "\n" +
            "".join([f"{chr(65+i)}. {c}\n" for i,c in enumerate(choices)]) + "Answer:")
        # score each option by negative NLL
        scores = []
        for i,c in enumerate(choices):
            inp = tokenizer(base + f" {chr(65+i)}", return_tensors="pt",
                             truncation=True, padding=True, max_length=2048).to(device)
            lbl = inp.input_ids.clone()
            out = llm(**inp, labels=lbl)
            nll = out.loss * lbl.size(1)
            scores.append(-nll.item())
        pred = chr(65 + int(torch.argmax(torch.tensor(scores))))
        if pred == answer:
            correct += 1
        total += 1
    acc = correct / total if total>0 else 0.0
    print(f"→ Validation accuracy: {acc:.4f} ({correct}/{total})")

evaluate()
# === Training loop with evaluation ===
print("🚀 Starting encoder training...")
ACCUMULATION_STEPS = 64
TOTAL_EXAMPLES = len(train_raw)
HALF_STEP = TOTAL_EXAMPLES // 2

for epoch in range(1, EPOCHS + 1):
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    encoder_optimizer.zero_grad()
    
    for step, ex in enumerate(loop, start=1):
        question = ex['question']
        # 1) retrieve top-K
        docs = [d.page_content for d in vector_db.similarity_search(query=question, k=K)]
        # 2) CE losses
        correct_index = ord(ex['answer']) - ord("A")
        prompt = (
            f"{ex['question']}\n" +
            "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, ex["choices"])]) +
            "Answer:"
        )
        options = [f" {letter}" for letter in LETTER_INDICES[:len(ex["choices"])]]
        ce_losses = get_ce_losses_for_docs(prompt, correct_index, options, docs)
        # 3) rank
        sorted_idxs = sorted(range(K), key=lambda i: ce_losses[i])
        # 4) embeddings
        q_enc = tokenizer_enc(
            question, return_tensors="pt",
            truncation=True, padding=True,
            max_length=CHUNK_SIZE
        ).to(device)
        q_emb = encoder(**q_enc).last_hidden_state[:, 0]
        
        d_embs = torch.cat([
            encoder(**tokenizer_enc(
                d, return_tensors="pt",
                truncation=True, padding=True,
                max_length=CHUNK_SIZE
            ).to(device)).last_hidden_state[:, 0]
            for d in docs
        ], dim=0)
        sims = F.cosine_similarity(q_emb.expand(K, -1), d_embs)
        # 5) hinge loss
        loss = sum(F.relu(MARGIN + sims[j] - sims[i])
                   for i, j in zip(sorted_idxs, sorted_idxs[1:]))
        loss = loss / max(len(sorted_idxs) - 1, 1)
        
        # **scale down loss** so the accumulated gradient magnitude stays roughly the same
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        # only step & zero gradients every ACCUMULATION_STEPS
        if step % ACCUMULATION_STEPS == 0:
            encoder_optimizer.step()
            encoder_optimizer.zero_grad()
        
        loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)  # show un-scaled loss

        if step == HALF_STEP:
                # flush any pending gradients
                if step % ACCUMULATION_STEPS != 0:
                    encoder_optimizer.step()
                    encoder_optimizer.zero_grad()
                
                # evaluate & save half-epoch checkpoint
                print(f"\n🔖 Half-epoch reached at step {step}. Running evaluation…")
                evaluate()
                mid_ckpt = f"./encoder_ckpt_epoch{epoch}_half"
                os.makedirs(mid_ckpt, exist_ok=True)
                encoder.save_pretrained(mid_ckpt)
                print(f"✅ Saved mid-epoch checkpoint to {mid_ckpt}\n")

    # if number of examples isn't divisible by ACCUMULATION_STEPS, do one final step
    if step % ACCUMULATION_STEPS != 0:
        encoder_optimizer.step()
        encoder_optimizer.zero_grad()

    # Evaluate after each epoch
    evaluate()
    ckpt_dir = f"./encoder_ckpt_epoch{epoch}"
    os.makedirs(ckpt_dir, exist_ok=True)
    encoder.save_pretrained(ckpt_dir)
    print(f"✅ Saved encoder checkpoint to {ckpt_dir}")

# === Save encoder ===
os.makedirs("./encoder_finetuned", exist_ok=True)
encoder.save_pretrained("./encoder_finetuned")
print("✅ Encoder training and evaluation complete.")
