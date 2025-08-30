# preproccessing/prepare_latents_owl.py

import os
import sys
import glob
import random
import shutil
import argparse
import multiprocessing as mp
from functools import partial

import torch
import numpy as np
import yaml
from tqdm import tqdm
import torchvision.transforms.functional as TF

# Add project root to Python path to allow importing from owl-vaes
sys.path.append("./owl-vaes")

# Imports from your custom owl-vaes codebase
from owl_vaes.configs import Config as OwlVaeConfig
from owl_vaes.models import get_model_cls as get_owl_model_cls
from owl_vaes.utils import versatile_load
from owl_vaes.models.dcae import DCAE


def create_split(data_dir1, data_dir2, output_dir, split=0.8):
    output_dir = os.path.abspath(output_dir)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Collect files from each directory separately to ensure balanced distribution
    p1_files = sorted(glob.glob(os.path.join(data_dir1, "*.npz")))
    p2_files = sorted(glob.glob(os.path.join(data_dir2, "*.npz")))
    
    print(f"Found {len(p1_files)} P1_WIN files and {len(p2_files)} P2_WIN files")
    
    # Shuffle each group separately
    random.shuffle(p1_files)
    random.shuffle(p2_files)
    
    # Split each group proportionally
    num_p1_train = int(len(p1_files) * split)
    num_p2_train = int(len(p2_files) * split)
    
    train_files = p1_files[:num_p1_train] + p2_files[:num_p2_train]
    val_files = p1_files[num_p1_train:] + p2_files[num_p2_train:]
    
    # Shuffle the combined train and val sets
    random.shuffle(train_files)
    random.shuffle(val_files)
    
    print(f"Train split: {num_p1_train} P1_WIN + {num_p2_train} P2_WIN = {len(train_files)} total")
    print(f"Val split: {len(p1_files) - num_p1_train} P1_WIN + {len(p2_files) - num_p2_train} P2_WIN = {len(val_files)} total")

    # Copy train files
    print(f"Copying {len(train_files)} training files to {train_dir}")
    for f in tqdm(train_files, desc="Copying train files"):
        shutil.copy(f, os.path.join(train_dir, os.path.basename(f)))

    # Copy val files
    print(f"Copying {len(val_files)} validation files to {val_dir}")
    for f in tqdm(val_files, desc="Copying val files"):
        shutil.copy(f, os.path.join(val_dir, os.path.basename(f)))

    print("\n--- Data Splitting Complete ---")
    print(f"Train data saved in: {train_dir}")
    print(f"Validation data saved in: {val_dir}")


def load_owl_vae(ckpt_dir, device=None):
    """Loads a custom owl-vae model from a checkpoint directory."""
    print(f"Loading owl-vae checkpoint from {ckpt_dir}...")

    # Find the .yml config file inside the checkpoint directory
    yaml_files = [f for f in os.listdir(ckpt_dir) if f.endswith(('.yml', '.yaml'))]
    if not yaml_files:
        raise FileNotFoundError(f"No .yml or .yaml config file found in directory: {ckpt_dir}")
    config_path = os.path.join(ckpt_dir, yaml_files[0])

    # Find the .pt checkpoint file
    pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint file found in directory: {ckpt_dir}")
    ckpt_pt_path = os.path.join(ckpt_dir, pt_files[0])

    print(f"Using config: {config_path}")
    print(f"Using checkpoint: {ckpt_pt_path}")

    # Load model using owl-vae logic
    cfg = OwlVaeConfig.from_yaml(config_path).model
    vae = get_owl_model_cls(cfg.model_id)(cfg)
    state_dict = versatile_load(ckpt_pt_path)
    vae.load_state_dict(state_dict)

    # Move to device and set to eval mode
    vae.to(device)
    vae.eval()
    
    num_params = sum(p.numel() for p in vae.parameters())
    print(f"✅ VAE loaded from {ckpt_dir} on {device} with {num_params/1e6:.1f}M parameters.")
    return vae


def process_pose_data(pose_images, end_idx):
    """Process pose images with Gaussian blur."""
    processed_poses = []
    for frame_idx in range(end_idx):
        pose_frame_raw = pose_images[frame_idx]
        pose_frame = torch.from_numpy(pose_frame_raw).float()
        
        # If pose has 3 channels, take max across channels and keep 1 channel
        if pose_frame.ndim == 3 and pose_frame.shape[0] == 3:
            pose_frame = torch.max(pose_frame, dim=0, keepdim=True)[0]
        
        # Apply Gaussian blur
        pose_frame = TF.gaussian_blur(pose_frame.unsqueeze(0), kernel_size=3).squeeze(0)
        processed_poses.append(pose_frame)
    
    return torch.stack(processed_poses, dim=0)


def load_pose_data(pose_dir, round_name, end_idx, gpu_id):
    """Load and process pose data if available."""
    if not pose_dir:
        return None
        
    pose_file_path = os.path.join(pose_dir, f"{round_name}_two_player_poses.npz")
    if not os.path.exists(pose_file_path):
        print(f"GPU {gpu_id}: Warning - pose file not found: {pose_file_path}")
        return None
        
    pose_data = np.load(pose_file_path)
    if 'pose_images' not in pose_data:
        print(f"GPU {gpu_id}: Warning - 'pose_images' key not found in {pose_file_path}")
        pose_data.close()
        return None
        
    pose_images = pose_data['pose_images'][:end_idx]
    pose_data.close()
    
    pose_channel = process_pose_data(pose_images, end_idx)
    print(f"GPU {gpu_id}: Loaded and processed pose data for {round_name}, shape: {pose_channel.shape}")
    return pose_channel


def process_single_file_owl(args):
    """Process a single .npz file on a specific GPU using an owl-vae model."""
    npz_path, split_output_dir, vae_ckpt_dir, gpu_id, batch_size, pose_dir = args
    
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    
    round_name = os.path.splitext(os.path.basename(npz_path))[0]
    
    try:
        vae = load_owl_vae(vae_ckpt_dir, device=device)
        print(f"GPU {gpu_id}: Processing {round_name}...")

        # Create output directories
        for data_type in ["latents", "actions", "states"]:
            os.makedirs(os.path.join(split_output_dir, round_name, data_type), exist_ok=True)

        # Load and prepare data
        data = np.load(npz_path)
        mask = data['valid_frames']
        end_idx = int(np.where(mask == 1)[0][-1])
        images = torch.from_numpy(data['images'][:end_idx]).float()
        actions = data['actions_p1'][:end_idx]
        states = data['states'][:end_idx]
        data.close()

        # Load pose data if available
        pose_channel = load_pose_data(pose_dir, round_name, end_idx, gpu_id)

        # Process images in batches
        all_latents = []
        num_batches = (images.shape[0] + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            batch_end_idx = min((i + 1) * batch_size, images.shape[0])
            batch_images = images[start_idx:batch_end_idx]
            
            # Concatenate pose channel if available
            if pose_channel is not None:
                batch_pose = pose_channel[start_idx:batch_end_idx]
                batch_images = torch.cat([batch_images, batch_pose], dim=1)
            
            with torch.no_grad(), torch.amp.autocast('cuda', torch.bfloat16):
                batch_images = batch_images.to(device)
                batch_images = (batch_images / 255.0) * 2 - 1
                
                # Handle variable number of return values from encoder
                encoder_output = vae.encoder(batch_images)
                if isinstance(encoder_output, tuple):
                    mu = encoder_output[0]
                else:
                    mu = encoder_output
                    
                batch_latents = mu.to(torch.float16).cpu().numpy()
                all_latents.append(batch_latents)
            
            # Clean up GPU memory
            del batch_images, mu, batch_latents
            if pose_channel is not None:
                del batch_pose
            torch.cuda.empty_cache()

        final_latents = np.concatenate(all_latents, axis=0)
        final_latents = np.transpose(final_latents, (1, 0, 2, 3))

        # Save the processed data
        np.save(os.path.join(split_output_dir, round_name, "latents", f"{round_name}_latents.npy"), final_latents)
        np.save(os.path.join(split_output_dir, round_name, "actions", f"{round_name}_actions.npy"), actions)
        np.save(os.path.join(split_output_dir, round_name, "states", f"{round_name}_states.npy"), states)
        
        print(f"GPU {gpu_id}: Finished processing {round_name}")
        return round_name
        
    except Exception as e:
        import traceback
        print(f"GPU {gpu_id}: Error processing {round_name}: {str(e)}")
        traceback.print_exc()
        return f"ERROR: {round_name}"
        
    finally:
        torch.cuda.empty_cache()


import random # Make sure to add this import at the top of your file

def process_data_owl(data_dir, output_dir, vae_ckpt_dir, num_gpus, batch_size, pose_dir=None):
    print("--- Starting Multi-GPU Data Processing for OWL-VAE ---")
    
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        print(f"Warning: Only {available_gpus} GPUs available, requested {num_gpus}")
        num_gpus = available_gpus

    # 1. Create a single list to hold all file processing arguments
    all_file_args_with_info = []
    
    # 2. Loop through splits to gather all files into the master list
    for split in ['train', 'val']:
        split_data_dir = os.path.join(data_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        
        if not os.path.exists(split_data_dir):
            continue
            
        npz_files = sorted(glob.glob(os.path.join(split_data_dir, "*.npz")))
        if not npz_files:
            continue

        print(f"Found {len(npz_files)} rounds in '{split}' split.")

        # Add arguments for each file to the master list
        for npz in npz_files:
            # We bundle all necessary info for each file
            all_file_args_with_info.append((npz, split_output_dir, vae_ckpt_dir, batch_size, pose_dir))

    # 3. (Recommended) Shuffle the list to better balance the load
    # This prevents all large files from being processed on the same GPUs.
    random.shuffle(all_file_args_with_info)
    
    # 4. Add the final GPU ID to each argument tuple using round-robin assignment
    final_file_args = [
        (*file_info, i % num_gpus) 
        for i, file_info in enumerate(all_file_args_with_info)
    ]
    
    # Reorder args to match the expected (npz_path, split_output_dir, vae_ckpt_dir, gpu_id, batch_size, pose_dir)
    # The new order after adding gpu_id at the end is (npz, output, ckpt, batch, pose, gpu_id)
    # We need to re-arrange it to match what process_single_file_owl expects
    final_file_args = [
        (npz_path, split_dir, ckpt_dir, gpu_id, b_size, p_dir)
        for (npz_path, split_dir, ckpt_dir, b_size, p_dir, gpu_id) in final_file_args
    ]


    if not final_file_args:
        print("No files found to process. Exiting.")
        return

    # 5. Create a single pool and process the entire combined list
    print(f"\nProcessing a total of {len(final_file_args)} files across all splits...")
    with mp.Pool(processes=num_gpus) as pool:
        results = list(tqdm(pool.imap(process_single_file_owl, final_file_args), total=len(final_file_args), desc="Processing all files"))

    print(f"\nCompleted processing {len(results)} total files.")
    print("\n--- Multi-GPU OWL-VAE Data Processing Complete ---")

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Preprocess Tekken data with a custom OWL-VAE encoder.')
    parser.add_argument('--vae-ckpt-dir', type=str, required=True, 
                       help='Path to the custom OWL-VAE checkpoint directory (containing .pt and .yml files).')
    parser.add_argument('--data-dir', type=str, default="preproccessing/data_v3", 
                       help='Path to the directory containing train/val splits of raw .npz data.')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Output directory for the cached latent data.')
    parser.add_argument('--pose-dir', type=str, default=None, 
                       help='Path to directory containing pose npz files with pose_images key.')
    parser.add_argument('--num-gpus', type=int, default=None, 
                       help='Number of GPUs to use (default: auto-detect all).')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size for encoding frames on each GPU (default: 16).')
    
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
    # create_split('/home/sky/owl-wms/preproccessing/t3_pose/P1_WIN', '/home/sky/owl-wms/preproccessing/t3_pose/P1_WIN', '/home/sky/owl-wms/preproccessing/data_v3_pose', 0.95)