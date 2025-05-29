import os
import sys
from datasets import load_dataset, DatasetDict


def main():
    # Load the source dataset from the Hub
    print("Loading dataset 'igzi/pile-stem-corpus-small-semantic'...")
    dataset: DatasetDict = load_dataset("igzi/pile-stem-corpus-small-semantic")
    print(f"Loaded splits: {list(dataset.keys())}")

    # Push to a new repo under the same format
    target_repo = "igzi/MNLP_M2_document_repo"
    print(f"Pushing dataset to '{target_repo}'...")
    dataset.push_to_hub(
        repo_id=target_repo,
        private=False  # Set True if you want a private dataset
    )
    print("Push complete!")


if __name__ == "__main__":
    main()