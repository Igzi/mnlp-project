from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
from transformers import AutoTokenizer

# === Config ===
repo_id = "igzi/finetuned_4_options_full_dataset_rag"
checkpoint_path = "./finetuned_4_options_full_dataset_rag/checkpoint-259"

api = HfApi()
try:
    create_repo(repo_id, exist_ok=True, private=False)
except Exception as e:
    print("Repo may already exist:", e)

# === Upload model files ===
upload_folder(
    folder_path=checkpoint_path,
    repo_id=repo_id,
    path_in_repo="",  # root
    commit_message="Upload fine-tuned model checkpoint 259",
)

# Upload to your fine-tuned repo
upload_folder(
    folder_path="./qwen-tokenizer",
    repo_id=repo_id,
    path_in_repo="",
    commit_message="Add tokenizer files from base model",
)
