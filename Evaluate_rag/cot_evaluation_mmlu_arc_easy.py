import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re

# ------------ CONFIG ------------
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET = "allenai/ai2_arc"
SPLIT = "test"
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
    "Question: {question}\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
    "Let's think step by step.\n"
)

# ------------ LOAD DATASET ------------
ds = load_dataset(DATASET, "ARC-Easy", split=SPLIT).shuffle(42)
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

for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="Evaluating"):
    batch = [ds[j] for j in range(i, min(i+BATCH_SIZE, len(ds)))]
    prompts = [
        COT_PROMPT.format(
            question=ex["question"],
            A=ex["choices"]["text"][0],
            B=ex["choices"]["text"][1],
            C=ex["choices"]["text"][2],
            D=ex["choices"]["text"][3],
        )
        for ex in batch
    ]
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
        gt = LETTER_INDICES[ex["answerKey"]]
        result = {"question": ex["question"], "gt": gt, "pred": pred, "output": output}

        if pred is not None:
            if pred == gt:
                correct += 1
        else:
            no_answer += 1
            # Fallback to direct answer prediction
            # Format the prompt for direct answer
            direct_prompt = (
                "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n"
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
with open(f"qwen3_cot_arc_easy_{SPLIT}_results.json", "w") as f:
    json.dump(results, f, indent=2)
