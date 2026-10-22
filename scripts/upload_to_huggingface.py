#!/usr/bin/env python3
"""
Upload AutoVoice DistilBERT NER Model to Hugging Face Model Hub.

Usage:
  # 1. Login with your Hugging Face write token:
  uv run --with huggingface_hub huggingface-cli login

  # 2. Upload the model folder:
  uv run --with huggingface_hub python scripts/upload_to_huggingface.py --repo-id your-username/autovoice-automotive-ner
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload AutoVoice NER model to Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID in the format 'username/repo-name' (e.g. devdasrahul/autovoice-automotive-ner)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/ner",
        help="Path to local NER model directory (default: models/ner)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token (defaults to HF_TOKEN env var or stored login credential)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hugging Face repository private (default: public)",
    )

    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Run with: uv run --with huggingface_hub python scripts/upload_to_huggingface.py --repo-id <repo_id>")
        sys.exit(1)

    model_path = Path(args.model_dir).resolve()
    if not model_path.exists():
        print(f"Error: Model directory not found at {model_path}")
        sys.exit(1)

    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    for req in required_files:
        if not (model_path / req).exists():
            print(f"Error: Missing required model file: {req} in {model_path}")
            sys.exit(1)

    api = HfApi(token=args.token)

    print(f"Ensuring repository exists: {args.repo_id} (private={args.private})...")
    create_repo(
        repo_id=args.repo_id,
        token=args.token,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading files from {model_path} to https://huggingface.co/{args.repo_id}...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=args.repo_id,
        repo_type="model",
        token=args.token,
        commit_message="Upload AutoVoice fine-tuned DistilBERT automotive NER model and model card",
    )

    print("\n" + "=" * 60)
    print(" Upload Successful!")
    print(f" Model Hub URL: https://huggingface.co/{args.repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
