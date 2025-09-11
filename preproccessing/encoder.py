import torch
import gc
import time
import argparse
import json
import yaml
import os
import sys

# --- Add project root to Python path ---
# This allows us to import from the 'owl-vaes' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports from your custom owl-vaes codebase ---
from owl_vaes.configs import Config as OwlVaeConfig
from owl_vaes.models import get_model_cls as get_owl_model_cls
from owl_vaes.utils import versatile_load

# --- Imports from diffusers library ---
from diffusers import AutoencoderKLLTXVideo, AutoencoderKLWan

# Default device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_vae(ckpt_dir, dtype=torch.bfloat16, device=None):
    """
    Loads a VAE, automatically detecting and handling both diffusers models
    and custom owl-vaes checkpoints.
    """
    config_json_path = os.path.join(ckpt_dir, "config.json")

    # --- Case 1: It's a diffusers model (has config.json) ---
    if os.path.exists(config_json_path):
        print(f"Detected diffusers model in {ckpt_dir}. Loading...")
        with open(config_json_path, 'r') as f:
            config_data = json.load(f)
        
        class_name = config_data.get("_class_name")
        if class_name == "AutoencoderKLWan":
            autoencoder_cls = AutoencoderKLWan
        elif class_name == "AutoencoderKLLTXVideo":
            autoencoder_cls = AutoencoderKLLTXVideo
        else:
            raise ValueError(f"Unsupported diffusers VAE class in config.json: {class_name}")

        vae = autoencoder_cls.from_pretrained(ckpt_dir, torch_dtype=dtype)

    # --- Case 2: It's a custom owl-vae model ---
    else:
        print(f"Did not find config.json. Assuming owl-vae checkpoint in {ckpt_dir}. Loading...")
        
        # Find the .yml config file inside the checkpoint directory
        yaml_files = [f for f in os.listdir(ckpt_dir) if f.endswith(('.yml', '.yaml'))]
        if not yaml_files:
            raise FileNotFoundError(f"No .yml or .yaml config file found inside the owl-vae checkpoint directory: {ckpt_dir}")
        config_path = os.path.join(ckpt_dir, yaml_files[0])

        # Find the checkpoint file (.pt)
        if ckpt_dir.endswith('.pt'):
             ckpt_pt_path = ckpt_dir
        else:
            pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
            if not pt_files:
                raise FileNotFoundError(f"No .pt checkpoint file found in directory: {ckpt_dir}")
            ckpt_pt_path = os.path.join(ckpt_dir, pt_files[0])

        print(f"Using config: {config_path}")
        print(f"Using checkpoint: {ckpt_pt_path}")

        # Load model using the logic from owl_vae_bridge
        cfg = OwlVaeConfig.from_yaml(config_path).model
        vae = get_owl_model_cls(cfg.model_id)(cfg)
        state_dict = versatile_load(ckpt_pt_path)
        vae.load_state_dict(state_dict)

    # Move to device and print info
    vae.to(device)
    print(f"✅ VAE loaded from {ckpt_dir} on {device}.")
    num_params = sum(p.numel() for p in vae.parameters())
    print(f"✅ VAE loaded with {num_params/1e6:.1f}M parameters.")
    return vae


def load_encoder(ckpt_dir, device):
    """Loads the specified VAE and isolates its encoder."""
    print(f"Loading VAE from {ckpt_dir}...")
    vae = load_vae(ckpt_dir, device=device, dtype=torch.bfloat16)

    # Isolate the encoder
    encoder = vae.encoder.to(device).eval()

    # Clean up memory
    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Freeze the encoder's parameters
    for param in encoder.parameters():
        param.requires_grad = False

    print("\n✅ VAE Encoder is now isolated in memory.")
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"✅ VAE Encoder loaded with {num_params/1e6:.1f}M parameters.")
    
    return encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Encoder Isolation/Loading Script")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the VAE checkpoint directory.")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to load the VAE encoder on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()
    
    encoder = load_encoder(args.ckpt_dir, args.device)
    print("Encoder loaded successfully.")