import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re
import os
import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# ------------ CONFIG ------------
MODEL_NAME = "igzi/finetuned_10_options_rag_new"
DATASET = "igzi/nlp4education"
SPLIT = "train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 16  # Try 8, 16, or more depending on GPU

# ------------ LOAD MODEL & TOKENIZER ------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
model.eval()

# ------------ CHAIN OF THOUGHT PROMPT ------------
COT_PROMPT = (
    "You are a STEM expert. Solve the following multiple choice question step by step. "
    "After your reasoning, print only the final answer on a new line in the format 'Final Answer: X', "
    "where X is A, B, C, or D.\n"
    "{context}"
    "Question: {question}\n"
)

# ------------ LOAD DATASET ------------
ds = load_dataset(DATASET, split=SPLIT).shuffle(42)
print(f"Loaded {len(ds)} samples from {DATASET} ({SPLIT})")

# ------------ ANSWER EXTRACTION ------------
def extract_letter(output):
    m = re.search(r"Final Answer:\s*\(?([A-D])\)?", output, re.IGNORECASE)
    return m.group(1).upper() if m else None

LETTER_INDICES = ["A", "B", "C", "D"]
results = []
correct = 0
no_answer = 0
fallback_correct = 0

out_dir = "../../lighteval-clean/lighteval-epfl-mnlp/community_tasks-Copy1/faiss_vector_db2"  # Directory for the saved index
index_path = os.path.join(out_dir, "index.faiss")
docstore_path = os.path.join(out_dir, "docstore.pkl")
mapping_path = os.path.join(out_dir, "index_to_docstore_id.pkl")

model_kwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", # Dynamically check cuda
}
encode_kwargs = {
    "normalize_embeddings": True
}
embedding_model = HuggingFaceEmbeddings(
            model_name="igzi/MNLP_M2_document_encoder",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

# 1. Check if the files exist
if os.path.exists(index_path) and os.path.exists(docstore_path) and os.path.exists(mapping_path):
    index = faiss.read_index(index_path)
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)
    with open(mapping_path, "rb") as f:
        index_to_docstore_id = pickle.load(f)
    vector_db = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        distance_strategy=DistanceStrategy.COSINE,
        normalize_L2=True
    )

for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="Evaluating"):
    batch = [ds[j] for j in range(i, min(i+BATCH_SIZE, len(ds)))]
    prompts = []
    contexts = []
    for ex in batch:
        choices = ex["choices"]
        retrieved_docs = vector_db.similarity_search(
            query=f"{ex['question']}\n" +
            "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)]) +
            "Answer:", k=3)
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
        context = "\nRelavent Documents:\n"
        context += "\n\n".join([
            f"Document {str(i)}:::\n" + doc
            for i, doc in enumerate(retrieved_docs_text)
        ])
        context += "\n\n"
        contexts.append(context)
        prompts.append(COT_PROMPT.format(
            context=context,
            question=ex["question"]
        )+"".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)])+"\nLet's think step by step.\n")
    
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    for j, ex in enumerate(batch):
        # Get the generated output after the prompt
        generated = output_ids[j][inputs["input_ids"].shape[1]:]
        output = tokenizer.decode(generated, skip_special_tokens=True)
        pred = extract_letter(output)
        gt = LETTER_INDICES[ex["answer"]]
        result = {"question": ex["question"], "gt": gt, "pred": pred, "output": output}

        if pred is not None:
            if pred == gt:
                correct += 1
        else:
            no_answer += 1
            # Fallback to direct answer prediction
            # Format the prompt for direct answer
            direct_prompt = (
                "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n"+contexts[j]+
                f"{ex['question']}\n" +
                "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, ex["choices"])]) +
                "Answer:"
            )
            input_ids = tokenizer(direct_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(
                    **input_ids,
                    return_dict=True,
                )
                # Get logits for the next token after the prompt
                logits = outputs.logits[0, -1]  # shape: [vocab_size]
                answer_tokens = [tokenizer(f" {l}", add_special_tokens=False)["input_ids"][0] for l in LETTER_INDICES]
                answer_probs = logits[answer_tokens].softmax(dim=0)
                answer_idx = answer_probs.argmax().item()
                fallback_pred = LETTER_INDICES[answer_idx]
                result.update({
                    "fallback_pred": fallback_pred,
                    "fallback_probs": answer_probs.tolist(),
                })
                if fallback_pred == gt:
                    fallback_correct += 1

        results.append(result)

# ------------ FINAL ACCURACY ------------
total = len(ds)
cot_acc = correct / total
combined_acc = (correct + fallback_correct) / total

print(f"\nChain-of-Thought Accuracy on {DATASET} [{SPLIT}]: {cot_acc*100:.2f}%")
print(f"Combined Accuracy (with fallback): {combined_acc*100:.2f}%")
print(f"CoT answer extraction failed on {no_answer}/{total} samples")

# Optionally: save results for later inspection
import json
with open(f"qwen3_cot_nlp4edu_{SPLIT}_results.json", "w") as f:
    json.dump(results, f, indent=2)
