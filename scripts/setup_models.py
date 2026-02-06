#!/usr/bin/env python
"""
PePath Model Setup Script

This script helps users set up the foundation models required by PePath.
It handles Hugging Face authentication and model download.

Usage:
    python scripts/setup_models.py
    python scripts/setup_models.py --model virchow2
    python scripts/setup_models.py --all
"""

import argparse
import os
import sys
from pathlib import Path

# Model information
MODELS = {
    "virchow2": {
        "name": "Virchow2",
        "hub_id": "paige-ai/Virchow2",
        "url": "https://huggingface.co/paige-ai/Virchow2",
        "description": "Paige AI pathology foundation model (default)",
        "embedding_dim": 2560,
    },
    "uni2": {
        "name": "UNI2",
        "hub_id": "MahmoodLab/UNI2-h",
        "url": "https://huggingface.co/MahmoodLab/UNI2-h",
        "description": "Mahmood Lab universal pathology model",
        "embedding_dim": 1536,
    },
    "gigapath": {
        "name": "GigaPath",
        "hub_id": "prov-gigapath/prov-gigapath",
        "url": "https://huggingface.co/prov-gigapath/prov-gigapath",
        "description": "Providence large-scale pathology model",
        "embedding_dim": 1536,
    },
}


def print_header():
    print("\n" + "=" * 60)
    print("🔬 PePath Model Setup")
    print("=" * 60 + "\n")


def check_huggingface_auth():
    """Check if user is authenticated with Hugging Face."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
        return True
    except Exception:
        return False


def login_huggingface():
    """Prompt user to login to Hugging Face."""
    print("\n📋 Hugging Face Authentication Required")
    print("-" * 40)
    print("You need to login to Hugging Face to download the models.")
    print("Please run: huggingface-cli login")
    print("\nOr set the HF_TOKEN environment variable.")
    
    response = input("\nWould you like to login now? [y/N]: ").strip().lower()
    if response == 'y':
        try:
            from huggingface_hub import login
            login()
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False
    return False


def check_model_access(model_key):
    """Check if user has access to a model."""
    model_info = MODELS[model_key]
    try:
        from huggingface_hub import model_info as hf_model_info
        info = hf_model_info(model_info["hub_id"])
        return True
    except Exception as e:
        if "401" in str(e) or "403" in str(e):
            print(f"⚠ No access to {model_info['name']}.")
            print(f"  Please accept the license at: {model_info['url']}")
            return False
        return True  # Assume accessible if error is not auth-related


def download_model(model_key):
    """Download and cache a model."""
    model_info = MODELS[model_key]
    print(f"\n📥 Downloading {model_info['name']}...")
    print(f"   Hub ID: {model_info['hub_id']}")
    
    try:
        import timm
        import torch
        
        # Special handling for different models
        if model_key == "virchow2":
            from timm.layers import SwiGLUPacked
            model = timm.create_model(
                f"hf-hub:{model_info['hub_id']}",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU
            )
        else:
            model = timm.create_model(
                f"hf-hub:{model_info['hub_id']}",
                pretrained=True
            )
        
        print(f"✓ {model_info['name']} downloaded and cached successfully!")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {model_info['name']}: {e}")
        return False


def list_models():
    """List available models."""
    print("\n📋 Available Models:")
    print("-" * 60)
    for key, info in MODELS.items():
        print(f"\n  {key}:")
        print(f"    Name: {info['name']}")
        print(f"    Embedding Dim: {info['embedding_dim']}")
        print(f"    Description: {info['description']}")
        print(f"    URL: {info['url']}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup foundation models for PePath"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(MODELS.keys()),
        help="Specific model to download"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all models"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check authentication and model access"
    )
    
    args = parser.parse_args()
    
    print_header()
    
    if args.list:
        list_models()
        return
    
    # Check authentication
    print("🔐 Checking Hugging Face authentication...")
    if not check_huggingface_auth():
        if not login_huggingface():
            print("\n⚠ Please login to Hugging Face before downloading models.")
            print("  Run: huggingface-cli login")
            sys.exit(1)
    
    if args.check:
        print("\n📋 Checking model access...")
        for key in MODELS:
            access = check_model_access(key)
            status = "✓" if access else "✗"
            print(f"  {status} {MODELS[key]['name']}")
        return
    
    # Download models
    models_to_download = []
    if args.all:
        models_to_download = list(MODELS.keys())
    elif args.model:
        models_to_download = [args.model]
    else:
        # Default: download virchow2
        print("\nNo model specified. Downloading default model (virchow2)...")
        models_to_download = ["virchow2"]
    
    success_count = 0
    for model_key in models_to_download:
        if check_model_access(model_key):
            if download_model(model_key):
                success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✓ Setup complete! {success_count}/{len(models_to_download)} models downloaded.")
    print("=" * 60 + "\n")
    
    print("Next steps:")
    print("  1. Run the demo: jupyter notebook examples/demo.ipynb")
    print("  2. Process slides: python -m pepath.cli --slide-path /path/to/slides")
    print()


if __name__ == "__main__":
    main()
