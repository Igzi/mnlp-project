from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModel

ckpt_dir = "./encoder_ckpt_epoch1"
repo_id  = "igzi/MNLP_document_encoder-finetuned"

# 1) Load your encoder weights
encoder = AutoModel.from_pretrained(ckpt_dir)

# 2) Reload & save the tokenizer into the same folder (if not already there)
tokenizer_enc = AutoTokenizer.from_pretrained("igzi/MNLP_M2_document_encoder", trust_remote_code=True)
tokenizer_enc.save_pretrained(ckpt_dir)

# 3) (Optional) programmatic login
#    Either run `huggingface-cli login` beforehand or:
# HfFolder.save_token(os.getenv("HF_TOKEN"))

# 4) Create the repo (if it doesn’t exist)
api = HfApi()
try:
    api.create_repo(repo_id=repo_id, token=True, private=False)
except Exception as e:
    print(f"⚠️  Repo might already exist: {e}")

# 5) Push both model and tokenizer
encoder.push_to_hub(repo_id, use_auth_token=True, commit_message="Upload fine-tuned encoder model")
tokenizer_enc.push_to_hub(repo_id, use_auth_token=True, commit_message="Upload fine-tuned encoder tokenizer")

print(f"✅ Encoder and tokenizer uploaded to https://huggingface.co/{repo_id}")
