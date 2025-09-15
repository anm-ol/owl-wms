import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from omegaconf import OmegaConf

# Add project root to Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append("./owl-vaes")

from owl_wms.configs import Config as ModelConfig
from owl_wms.models import get_model_cls
from owl_wms.sampling import get_sampler_cls
from owl_wms.data import get_loader as get_data_loader
from owl_wms.utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from owl_wms.utils.logging import process_video_frames

# Try to import moviepy, but make it optional
try:
    from moviepy.editor import ImageSequenceClip
    HAS_MOVIEPY = True
except ImportError:
    print("Warning: moviepy not available. Will save frames as numpy array instead.")
    HAS_MOVIEPY = False


def load_model(model_config_path, model_ckpt_path, device, compile_model=False):
    """
    Load and initialize the Tekken model.
    
    Args:
        model_config_path (str): Path to the model configuration file
        model_ckpt_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model on
        compile_model (bool): Whether to compile the model for faster inference
        
    Returns:
        model: Loaded and initialized model
        model_cfg: Model configuration
        train_cfg: Training configuration
    """
    print(f"Loading TekkenRFTv2 model from checkpoint: {model_ckpt_path}")
    
    # Load main model configuration
    model_cfg_main = ModelConfig.from_yaml(model_config_path)
    model_cfg = model_cfg_main.model
    train_cfg = model_cfg_main.train
    
    # Initialize model
    model = get_model_cls(model_cfg.model_id)(model_cfg).core
    
    # Load state dictionary and clean keys
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


def load_vae_decoder(vae_id, vae_cfg_path, vae_ckpt_path, vae_batch_size, device, compile_decoder=False):
    """
    Load and initialize the VAE decoder.
    
    Args:
        vae_id (str): VAE model ID
        vae_cfg_path (str): Path to VAE configuration
        vae_ckpt_path (str): Path to VAE checkpoint
        vae_batch_size (int): Batch size for VAE processing
        device (torch.device): Device to load the decoder on
        compile_decoder (bool): Whether to compile the decoder
        
    Returns:
        decode_fn: Batched decode function
    """
    print(f"Loading VAE decoder from checkpoint: {vae_ckpt_path}")
    
    decoder = get_decoder_only(vae_id, vae_cfg_path, vae_ckpt_path)
    decoder = decoder.to(device).eval()
    
    if compile_decoder:
        print("Compiling VAE decoder...")
        decoder = torch.compile(decoder, mode='max-autotune', fullgraph=True)
    
    decode_fn = make_batched_decode_fn(decoder, vae_batch_size, temporal_vae=False)
    print("VAE decoder loaded successfully.")
    return decode_fn


def load_data_and_actions(sample_data_id, batch_size, sample_data_kwargs, starting_frame_index, 
                         initial_context_length, actions_npy_path, vae_scale, device):
    """
    Load initial context data and action sequence.
    
    Args:
        sample_data_id (str): Sample data identifier
        batch_size (int): Batch size for data loading
        sample_data_kwargs (dict): Additional kwargs for data loader
        starting_frame_index (int): Index of starting frame in dataset
        initial_context_length (int): Length of initial context
        actions_npy_path (str): Path to actions numpy file
        vae_scale (float): VAE scaling factor
        device (torch.device): Device to load data on
        
    Returns:
        initial_latents: Initial latent representations
        actions: Action sequence tensor
    """
    print("Loading sample data for context...")
    
    sample_dataset = get_data_loader(
        sample_data_id,
        batch_size,
        **sample_data_kwargs
    ).dataset

    if starting_frame_index >= len(sample_dataset):
        raise ValueError(f"starting_frame_index {starting_frame_index} is out of bounds for dataset with length {len(sample_dataset)}")

    data_dict = sample_dataset[starting_frame_index]
    initial_latents = data_dict['latents'].unsqueeze(0).to(device).bfloat16() / vae_scale
    
    if initial_context_length:
        initial_latents = initial_latents[:, :initial_context_length]
    
    print(f"Loaded context from data window index {starting_frame_index}")
    
    print(f"Loading actions from '{actions_npy_path}'...")
    full_action_sequence = np.load(actions_npy_path)
    if full_action_sequence.ndim > 1:
        full_action_sequence = full_action_sequence[:, -1]
    actions = torch.from_numpy(full_action_sequence).to(device).long().unsqueeze(0)
    print(f"Loaded action sequence of length {actions.shape[1]}")
    
    return initial_latents, actions


def initialize_sampler(sampler_id, sampler_kwargs, num_frames):
    """
    Initialize the sampler for inference.
    
    Args:
        sampler_id (str): Sampler identifier
        sampler_kwargs (dict): Sampler configuration
        num_frames (int): Number of frames to generate
        
    Returns:
        sampler: Initialized sampler
    """
    print("Initializing sampler...")
    sampler_kwargs = dict(sampler_kwargs)  # Create a mutable copy
    sampler_kwargs['num_frames'] = num_frames
    sampler = get_sampler_cls(sampler_id)(**sampler_kwargs)
    print(f"Using sampler '{sampler_id}' to generate {num_frames} new frames.")
    return sampler


def generate_video_from_latents(model, sampler, initial_latents, actions, decode_fn, vae_scale):
    """
    Generate video output from initial latents and actions.
    
    Args:
        model: The loaded Tekken model
        sampler: Initialized sampler
        initial_latents: Initial latent representations
        actions: Action sequence tensor
        decode_fn: VAE decode function
        vae_scale (float): VAE scaling factor
        
    Returns:
        video_out: Generated video output
        out_actions: Output actions
        inference_time: Time taken for inference
    """
    print("Running inference...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        video_out, _, out_actions = sampler(
            model,
            initial_latents,
            actions,
            decode_fn=decode_fn,
            vae_scale=vae_scale
        )
    
    video_out = video_out.permute(0, 2, 1, 3, 4) 
    end_time.record()
    torch.cuda.synchronize()
    inference_time = start_time.elapsed_time(end_time) / 1000.0
    
    print(f"Inference completed in {inference_time:.2f} seconds.")
    return video_out, out_actions, inference_time


def save_video_output(video_out, out_actions, output_path):
    """
    Save the generated video output to file.
    
    Args:
        video_out: Generated video tensor
        out_actions: Output actions tensor
        output_path (str): Path to save the output video
    """
    if video_out is None:
        print("Inference did not produce a video output to save.")
        return

    print(f"Saving output video to '{output_path}'...")
    processed_frames = process_video_frames(video_out, out_actions, max_samples=4)
    
    if HAS_MOVIEPY:
        frame_list = []
        for i in range(processed_frames.shape[1]):
            frame = np.moveaxis(processed_frames[0, i], 0, -1) 
            frame_list.append(frame)
        
        try:
            video_clip = ImageSequenceClip(frame_list, fps=30)
            video_clip.write_videofile(
                output_path,
                fps=30,
                codec='libx264',
                verbose=False,
                logger=None
            )
            print("Video saved successfully.")
        except Exception as e:
            print(f"Error saving video with moviepy: {e}")
            print("Saving processed frames as numpy array...")
            np.save(output_path.replace('.mp4', '_frames.npy'), processed_frames)
            print(f"Frames saved as numpy array: {output_path.replace('.mp4', '_frames.npy')}")
    else:
        print("MoviePy not available. Saving processed frames as numpy array...")
        np.save(output_path.replace('.mp4', '_frames.npy'), processed_frames)
        print(f"Frames saved as numpy array: {output_path.replace('.mp4', '_frames.npy')}")


def run_inference(cfg):
    """
    Main function to run fast inference for the Tekken model based on a config object.
    """
    print("="*50)
    print("STARTING INFERENCE")
    print("="*50)
    print(f"Config: {cfg}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and configurations
    model, model_cfg, train_cfg = load_model(
        cfg.model_config_path,
        cfg.model_ckpt_path,
        device,
        cfg.compile
    )

    # Load VAE decoder
    decode_fn = load_vae_decoder(
        train_cfg.vae_id,
        train_cfg.vae_cfg_path,
        train_cfg.vae_ckpt_path,
        train_cfg.vae_batch_size,
        device,
        cfg.compile
    )

    # Load data and actions
    initial_latents, actions = load_data_and_actions(
        train_cfg.sample_data_id,
        cfg.batch_size,
        train_cfg.sample_data_kwargs,
        cfg.starting_frame_index,
        cfg.initial_context_length,
        cfg.actions_npy_path,
        train_cfg.vae_scale,
        device
    )

    # Initialize sampler
    sampler = initialize_sampler(
        train_cfg.sampler_id,
        train_cfg.sampler_kwargs,
        cfg.num_frames
    )

    # Generate video
    video_out, out_actions, inference_time = generate_video_from_latents(
        model,
        sampler,
        initial_latents,
        actions,
        decode_fn,
        train_cfg.vae_scale
    )

    # Save output
    save_video_output(video_out, out_actions, cfg.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast inference script for the Tekken model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yml",
        help="Path to the inference configuration YAML file."
    )
    args = parser.parse_args()
    
    # Load the YAML configuration file
    inference_cfg = OmegaConf.load(args.config)
    run_inference(inference_cfg)