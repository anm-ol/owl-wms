import argparse
import torch
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
from omegaconf import OmegaConf
import moviepy.editor as mp
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['TORCHDYNAMO_VERBOSE'] = '1'

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "owl-vaes"))

from owl_wms.configs import Config as ModelConfig
from owl_wms.models import get_model_cls
from owl_wms.sampling import get_sampler_cls
from owl_wms.utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from owl_wms.utils.ddp import setup, cleanup

def load_model(model_config_path, model_ckpt_path, device, compile_model=False):
    """
    Loads and initializes the Tekken model.
    """
    print(f"Loading TekkenRFTv2 model from checkpoint: {model_ckpt_path}")
    
    model_cfg_main = ModelConfig.from_yaml(model_config_path)
    model_cfg = model_cfg_main.model
    train_cfg = model_cfg_main.train
    
    model = get_model_cls(model_cfg.model_id)(model_cfg).core
    
    state_dict = torch.load(model_ckpt_path, map_location='cpu')
    cleaned_state_dict = {}
    prefix_to_strip = '_orig_mod.core.'
    for k, v in state_dict.items():
        if k.startswith(prefix_to_strip):
            new_k = k[len(prefix_to_strip):]
            cleaned_state_dict[new_k] = v
        else:
            cleaned_state_dict[k] = v
    
    model.load_state_dict(cleaned_state_dict)
    model = model.to(device).eval()
    
    if compile_model:
        print("Compiling model for faster inference...")
        model = torch.compile(model, mode='max-autotune', fullgraph=True)
    
    print("Model loaded successfully.")
    return model, model_cfg, train_cfg

def load_vae_decoder(train_cfg, device, compile_decoder=False):
    """
    Loads and initializes the VAE decoder from the training config.
    """
    print(f"Loading VAE decoder from checkpoint: {train_cfg.vae_ckpt_path}")
    
    decoder = get_decoder_only(train_cfg.vae_id, train_cfg.vae_cfg_path, train_cfg.vae_ckpt_path)
    decoder = decoder.to(device).eval()
    
    if compile_decoder:
        print("Compiling VAE decoder...")
        decoder = torch.compile(decoder, mode='max-autotune', fullgraph=True)
    
    decode_fn = make_batched_decode_fn(decoder, train_cfg.vae_batch_size, temporal_vae=True)
    print("VAE decoder loaded successfully.")
    return decode_fn

def load_round_data(round_dir, device):
    """Loads all data (latents, actions) for a single round."""
    round_name = os.path.basename(round_dir)
    latents_path = os.path.join(round_dir, 'latents', f'{round_name}_latents.npy')
    actions_path = os.path.join(round_dir, 'actions', f'{round_name}_actions.npy')

    if not all(os.path.exists(p) for p in [latents_path, actions_path]):
        print(f"Skipping incomplete round: {round_dir}")
        return None, None, None

    all_latents_npy = np.load(latents_path)
    all_actions_npy = np.load(actions_path)

    all_latents = torch.from_numpy(all_latents_npy).to(device).float()
    all_actions = torch.from_numpy(all_actions_npy).to(device).long()
    
    return all_latents, all_actions, round_name

def save_video_clip(frames_tensor, output_path, fps=30):
    """
    Saves a video from a PyTorch tensor using MoviePy.
    """
    if frames_tensor is None:
        print(f"Warning: No frames to save for {output_path}.")
        return

    video_np = frames_tensor.squeeze().permute(0, 2, 3, 1).cpu().numpy()
    video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)

    clip = mp.ImageSequenceClip(list(video_np), fps=fps)
    clip.write_videofile(output_path, codec="libx264")
    print(f"Video saved successfully to {output_path}.")

def pad_actions_batch(actions_list, pad_value=0):
    """
    Pad a list of action tensors to the same length.
    
    Args:
        actions_list (list): List of action tensors with potentially different lengths
        pad_value (int): Value to use for padding
        
    Returns:
        torch.Tensor: Padded actions tensor with consistent length
    """
    if not actions_list:
        return torch.empty(0)
    
    # Find the maximum length in the batch
    max_length = max(actions.shape[-1] if actions.dim() > 1 else actions.shape[0] for actions in actions_list)
    
    padded_actions = []
    for actions in actions_list:
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        current_length = actions.shape[1]
        if current_length < max_length:
            pad_length = max_length - current_length
            pad_tensor = torch.full((actions.shape[0], pad_length), pad_value, 
                                   dtype=actions.dtype, device=actions.device)
            actions = torch.cat([actions, pad_tensor], dim=1)
        elif current_length > max_length:
            actions = actions[:, :max_length]
        
        padded_actions.append(actions)
    
    return torch.cat(padded_actions, dim=0), max_length

def simulate_batch(model, sampler, decode_fn, train_cfg, batch_data, output_dir, max_generation_length=None):
    """
    Simulates a batch of rounds simultaneously for better GPU utilization.
    """
    batch_initial_latents = []
    batch_actions_list = []
    batch_round_names = []
    
    for initial_latents, actions, round_name in batch_data:
        if initial_latents.dim() == 4:
            initial_latents = initial_latents.unsqueeze(0).permute(0, 2, 1, 3, 4)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        batch_initial_latents.append(initial_latents)
        batch_actions_list.append(actions)
        batch_round_names.append(round_name)
    
    # Concatenate latents along batch dimension
    batch_initial_latents = torch.cat(batch_initial_latents, dim=0)
    
    # Pad actions to consistent length
    pad_value = getattr(train_cfg, 'action_pad_value', 0)
    batch_actions, max_action_len = pad_actions_batch(batch_actions_list, pad_value)
    
    # Apply max generation length if specified
    if max_generation_length is not None:
        initial_context_len = batch_initial_latents.shape[2]
        max_total_frames = initial_context_len + max_generation_length
        if batch_actions.shape[1] > max_total_frames:
            batch_actions = batch_actions[:, :max_total_frames]
            max_action_len = max_total_frames
    
    total_frames = batch_actions.shape[1]
    initial_context_len = batch_initial_latents.shape[2]
    sampler.num_frames = total_frames - initial_context_len
    
    normalized_initial_latents = batch_initial_latents / train_cfg.vae_scale
    
    print(f"Processing batch of {len(batch_data)} rounds with padded action length: {max_action_len}")
    if max_generation_length is not None:
        print(f"Generation limited to {sampler.num_frames} frames (max_generation_length: {max_generation_length})")
    

    # Generate videos for the entire batch
    with torch.no_grad(), torch.amp.autocast('cuda', torch.bfloat16): 
        generated_videos, _, _ = sampler(
            model,
            normalized_initial_latents,
            batch_actions,
            compile_on_decode=True,
            decode_fn=decode_fn,
        )
    
    # Generate ground truth videos for the batch
    ground_truth_latents = batch_actions[:, :generated_videos.shape[1]]
    ground_truth_videos = decode_fn(ground_truth_latents / train_cfg.vae_scale)
    
    # Save individual videos from the batch
    for i, round_name in enumerate(batch_round_names):
        gt_output_path = os.path.join(output_dir, f"{round_name}_ground_truth.mp4")
        save_video_clip(ground_truth_videos[i:i+1], gt_output_path)
        
        gen_output_path = os.path.join(output_dir, f"{round_name}_generated.mp4")
        save_video_clip(generated_videos[i:i+1], gen_output_path)

def simulate_round(model, sampler, decode_fn, train_cfg, initial_latents, actions, output_dir, round_name, max_generation_length=None):
    """
    Simulates a full round of video generation and saves the ground truth and generated videos separately.
    """
    print(f"Simulating round: {round_name}...")

    total_frames = actions.shape[0]
    initial_context_len = initial_latents.shape[1] if initial_latents.dim() == 5 else initial_latents.shape[0]
    
    # Apply max generation length if specified
    if max_generation_length is not None:
        max_total_frames = initial_context_len + max_generation_length
        if total_frames > max_total_frames:
            total_frames = max_total_frames
            actions = actions[:max_total_frames]
    
    sampler.num_frames = total_frames - initial_context_len

    if initial_latents.dim() == 4:
        initial_latents = initial_latents.unsqueeze(0).permute(0, 2, 1, 3, 4)
        actions = actions.unsqueeze(0)

    normalized_initial_latents = initial_latents / train_cfg.vae_scale

    if max_generation_length is not None:
        print(f"Generation limited to {sampler.num_frames} frames (max_generation_length: {max_generation_length})")

    with torch.no_grad(), torch.amp.autocast('cuda', torch.bfloat16): 
        generated_videos, _, _ = sampler(
            model,
            normalized_initial_latents,
            actions,
            compile_on_decode=True,
            decode_fn=decode_fn,
        )

    ground_truth_latents = actions[:,:generated_videos.shape[1]]
    ground_truth_videos = decode_fn(ground_truth_latents / train_cfg.vae_scale)

    gt_output_path = os.path.join(output_dir, f"{round_name}_ground_truth.mp4")
    save_video_clip(ground_truth_videos, gt_output_path)

    gen_output_path = os.path.join(output_dir, f"{round_name}_generated.mp4")
    save_video_clip(generated_videos, gen_output_path)

def run_simulation(cfg):
    """Main function to orchestrate the simulation pipeline, adapted for DDP with batch processing."""
    global_rank, local_rank, world_size = setup()
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    if global_rank == 0:
        print("="*50)
        print(f"STARTING DISTRIBUTED ROUND SIMULATION ON {world_size} GPUS")
        print("="*50)

    model, model_cfg, train_cfg = load_model(cfg.model_config_path, cfg.model_ckpt_path, device, cfg.compile)
    decode_fn = load_vae_decoder(train_cfg, device, cfg.compile)
    sampler = get_sampler_cls(train_cfg.sampler_id)(**train_cfg.sampler_kwargs)

    # Distribute work among processes
    data_dir = cfg.data_dir
    output_dir = cfg.output_dir
    batch_size = getattr(cfg, 'batch_size', 1)  # Default to 1 if not specified
    max_generation_length = getattr(cfg, 'max_generation_length', None)
    
    if global_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        if max_generation_length is not None:
            print(f"Maximum generation length set to: {max_generation_length} frames")
    
    # Synchronize after directory creation
    dist.barrier()
    
    all_round_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, 'round_*')) if os.path.isdir(d)])
    
    if not all_round_dirs:
        if global_rank == 0:
            print(f"No round directories found in {data_dir}. Exiting.")
        return

    # Shard the list of directories across the processes
    my_round_dirs = all_round_dirs[global_rank::world_size]

    if global_rank == 0:
        print(f"Found {len(all_round_dirs)} total rounds. Each GPU will process approximately {len(my_round_dirs)} rounds.")
        print(f"Using batch size: {batch_size}")
    
    # Process rounds in batches
    for i in tqdm(range(0, len(my_round_dirs), batch_size), desc=f"GPU {global_rank} Batch Progress", position=global_rank):
        batch_round_dirs = my_round_dirs[i:i + batch_size]
        batch_data = []
        
        # Load data for the current batch
        for round_dir in batch_round_dirs:
            latents, actions, round_name = load_round_data(round_dir, device)
            
            if latents is None or actions is None:
                continue
                
            initial_context_len = min(cfg.initial_context_length, latents.shape[1] if latents.dim() == 5 else latents.shape[0])
            initial_latents = latents[:, :initial_context_len]
            
            batch_data.append((initial_latents, actions, round_name))
        
        if not batch_data:
            continue
        
        # Process the batch
        if len(batch_data) == 1:
            # Single round - use original function for simplicity
            initial_latents, actions, round_name = batch_data[0]
            simulate_round(model, sampler, decode_fn, train_cfg, initial_latents, actions, output_dir, round_name, max_generation_length)
        else:
            # Multiple rounds - use batch processing
            simulate_batch(model, sampler, decode_fn, train_cfg, batch_data, output_dir, max_generation_length)
        
        # Clear GPU cache between batches
        torch.cuda.empty_cache()
    
    # Final synchronization
    dist.barrier()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate entire rounds of Tekken video generation from cached latents.")
    parser.add_argument("--config", type=str, default="configs/simulate_config.yml", help="Path to the simulation configuration YAML file.")
    args = parser.parse_args()

    # The main process is launched by torchrun, which handles multiprocessing setup.
    # The script can still be run locally without torchrun, but without DDP.
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    base_inference_cfg = OmegaConf.load(args.config)
    run_simulation(base_inference_cfg)