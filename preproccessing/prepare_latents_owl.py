# preproccessing/prepare_latents_owl.py (Optimized Version)

import os
import sys
import glob
import random
import shutil
import argparse
import traceback
import multiprocessing as mp

import torch
import numpy as np
import yaml
from tqdm import tqdm
import torchvision.transforms.functional as TF

# Add project root to Python path
sys.path.append("./owl-vaes")

# Imports from custom owl-vaes codebase
from owl_vaes.configs import Config as OwlVaeConfig
from owl_vaes.models import get_model_cls as get_owl_model_cls
from owl_vaes.utils import versatile_load

# --- REMOVED: Global variable approach that caused device conflicts ---
# _process_models = {}

def load_owl_vae(ckpt_dir, device=None):
    """Loads a custom owl-vae model from a checkpoint directory."""
    yaml_files = [f for f in os.listdir(ckpt_dir) if f.endswith(('.yml', '.yaml'))]
    if not yaml_files:
        raise FileNotFoundError(f"No config file found in {ckpt_dir}")
    config_path = os.path.join(ckpt_dir, yaml_files[0])

    pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint file found in {ckpt_dir}")
    ckpt_pt_path = os.path.join(ckpt_dir, pt_files[0])

    cfg = OwlVaeConfig.from_yaml(config_path).model
    vae = get_owl_model_cls(cfg.model_id)(cfg)
    state_dict = versatile_load(ckpt_pt_path)
    vae.load_state_dict(state_dict)

    vae.to(device)
    vae.eval()
    
    num_params = sum(p.numel() for p in vae.parameters())
    return vae

def _encode_batches(vae, images, pose_channel, batch_size, device):
    """Helper function to encode images in batches."""
    all_latents = []
    num_batches = (images.shape[0] + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, images.shape[0])
        batch_images = images[start_idx:end_idx]
        
        if pose_channel is not None:
            batch_pose = pose_channel[start_idx:end_idx]
            batch_images = torch.cat([batch_images, batch_pose], dim=1)
        
        with torch.no_grad(), torch.amp.autocast('cuda', torch.bfloat16):
            batch_images = batch_images.to(device)
            batch_images = (batch_images / 255.0) * 2 - 1
            
            encoder_output = vae.encoder(batch_images)
            mu = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
            
            batch_latents = mu.to(torch.float16).cpu().numpy()
            all_latents.append(batch_latents)
        
        del batch_images, mu, batch_latents
        if pose_channel is not None:
            del batch_pose
        torch.cuda.empty_cache()
        
    return np.concatenate(all_latents, axis=0)

def process_single_file_owl(args):
    """Process a single .npz file. Loads the VAE model once per process."""
    npz_path, split_output_dir, vae_ckpt_dir, gpu_id, batch_size, pose_dir = args
    
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    round_name = os.path.splitext(os.path.basename(npz_path))[0]

    try:
        # --- NEW: Load model fresh for each worker to avoid device conflicts ---
        # Check if we already have a model loaded for this process
        if not hasattr(process_single_file_owl, '_cached_model'):
            process_single_file_owl._cached_model = {}
        
        if gpu_id not in process_single_file_owl._cached_model:
            process_single_file_owl._cached_model[gpu_id] = load_owl_vae(vae_ckpt_dir, device=device)
        
        vae_model = process_single_file_owl._cached_model[gpu_id]
        
        # Create output directories
        os.makedirs(os.path.join(split_output_dir, round_name, "latents"), exist_ok=True)
        os.makedirs(os.path.join(split_output_dir, round_name, "actions"), exist_ok=True)
        os.makedirs(os.path.join(split_output_dir, round_name, "states"), exist_ok=True)
        
        # Load data from .npz
        data = np.load(npz_path)
        mask = data['valid_frames']
        end_idx = int(np.where(mask == 1)[0][-1])
        images = torch.from_numpy(data['images'][:end_idx]).float()
        actions = data['actions_p1'][:end_idx]
        states = data['states'][:end_idx]
        data.close()

        # Load pose data (if available)
        pose_channel = None
        if pose_dir:
            pose_file = os.path.join(pose_dir, f"{round_name}_two_player_poses.npz")
            if os.path.exists(pose_file):
                pose_data = np.load(pose_file)
                if 'pose_images' in pose_data:
                    pose_imgs = torch.from_numpy(pose_data['pose_images'][:end_idx]).float()
                    pose_channel = torch.max(pose_imgs, dim=1, keepdim=True)[0]
                    pose_channel = TF.gaussian_blur(pose_channel, kernel_size=3)
                pose_data.close()
        
        # Encode images to latents in batches
        final_latents = _encode_batches(vae_model, images, pose_channel, batch_size, device)
        
        # The transpose seems intentional for a downstream model expecting [C, T, H, W]
        final_latents = np.transpose(final_latents, (1, 0, 2, 3))
        
        # Save processed data
        np.save(os.path.join(split_output_dir, round_name, "latents", f"{round_name}_latents.npy"), final_latents)
        np.save(os.path.join(split_output_dir, round_name, "actions", f"{round_name}_actions.npy"), actions)
        np.save(os.path.join(split_output_dir, round_name, "states", f"{round_name}_states.npy"), states)
        
        return round_name

    except Exception as e:
        print(f"GPU {gpu_id}: FATAL ERROR processing {round_name}: {e}")
        traceback.print_exc()
        return f"ERROR: {round_name}"
    finally:
        torch.cuda.empty_cache()


def process_data_owl(data_dir, output_dir, vae_ckpt_dir, num_gpus, batch_size, pose_dir=None):
    """Main function to find files and distribute them to worker processes."""
    print("--- Starting Optimized Multi-GPU Data Processing ---")
    
    # 1. Gather all files from both 'train' and 'val' splits into a single list
    all_files_to_process = []
    for split in ['train', 'val']:
        split_data_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_data_dir):
            continue
        
        npz_files = glob.glob(os.path.join(split_data_dir, "*.npz"))
        split_output_dir = os.path.join(output_dir, split)
        
        for npz in npz_files:
            all_files_to_process.append({'path': npz, 'output': split_output_dir})
        print(f"Found {len(npz_files)} files in '{split}' split.")

    if not all_files_to_process:
        print("No .npz files found to process. Exiting.")
        return

    # 2. Shuffle for better load balancing
    random.shuffle(all_files_to_process)
    
    # 3. Prepare final arguments for each file, assigning a GPU ID in a round-robin fashion
    final_args = [
        (
            file_info['path'],
            file_info['output'],
            vae_ckpt_dir,
            i % num_gpus,
            batch_size,
            pose_dir
        )
        for i, file_info in enumerate(all_files_to_process)
    ]
    
    # 4. Process the entire list using the multiprocessing pool
    print(f"\nProcessing a total of {len(final_args)} files on {num_gpus} GPUs...")
    with mp.Pool(processes=num_gpus) as pool:
        results = list(tqdm(pool.imap(process_single_file_owl, final_args), total=len(final_args), desc="Encoding files"))

    print(f"\nCompleted processing {len(results)} files.")
    print("--- Multi-GPU Data Processing Complete ---")


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Preprocess data with a custom OWL-VAE encoder.')
    parser.add_argument('--vae-ckpt-dir', type=str, required=True, help='Path to the OWL-VAE checkpoint directory.')
    parser.add_argument('--data-dir', type=str, default="preproccessing/data_v3", help='Path to the directory with train/val splits.')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for cached latent data.')
    parser.add_argument('--pose-dir', type=str, default=None, help='Path to directory with pose npz files.')
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs to use (default: auto-detect all).')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for encoding frames on each GPU.')
    
    args = parser.parse_args()
    
    num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
    
    process_data_owl(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vae_ckpt_dir=args.vae_ckpt_dir,
        num_gpus=num_gpus,
        batch_size=args.batch_size,
        pose_dir=args.pose_dir
    )

    #python preproccessing/prepare_latents_owl.py --vae-ckpt-dir "preproccessing/checkpoints/tekken_vae_H200_v6" --data-dir "preproccessing/data_v3" --pose-dir "preproccessing/t3_pose" --output-dir "preproccessing/cached_dcae" --batch-size