import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import OmegaConf

# Add project root to Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append("./owl-vaes")
sys.path.append("./owl-vaes")

from owl_wms.configs import Config as ModelConfig
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
from owl_wms.utils.logging import process_video_frames

# Try to import moviepy, but make it optional
try:
    from moviepy.editor import ImageSequenceClip
    HAS_MOVIEPY = True
except ImportError:
    print("Warning: moviepy not available. Will save frames as numpy array instead.")
    HAS_MOVIEPY = False

def run_inference(cfg):
    """
    Main function to run fast inference for the Tekken model based on a config object.
    """
    print("="*50)
    print("STARTING INFERENCE")
    print("="*50)
    print(f"Config: {cfg}")
    
    # Load main model configuration from the path specified in the inference config
    print(f"DEBUG: Loading model config from: {cfg.model_config_path}")
    model_cfg_main = ModelConfig.from_yaml(cfg.model_config_path)
    model_cfg = model_cfg_main.model
    train_cfg = model_cfg_main.train
    
    print(f"DEBUG: Model config: {model_cfg}")
    print(f"DEBUG: Train config keys: {list(train_cfg.keys()) if hasattr(train_cfg, 'keys') else 'Not a dict'}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"DEBUG: CUDA device count: {torch.cuda.device_count()}")
        print(f"DEBUG: Current CUDA device: {torch.cuda.current_device()}")
        print(f"DEBUG: CUDA device name: {torch.cuda.get_device_name()}")

    # --- 1. Load Model ---
    print(f"Loading TekkenRFTv2 model from checkpoint: {cfg.model_ckpt_path}")
    # Initialize the .core of the model, as that's what we want to load weights into
    model = get_model_cls(model_cfg.model_id)(model_cfg).core
    print(f"Model core initialized.")
    print(f"DEBUG: Model type: {type(model)}")
    print(f"DEBUG: Model device before loading: {next(model.parameters()).device}")

    # Load the state dictionary from the checkpoint file.
    # This checkpoint is expected to be the EMA weights dictionary.
    print(f"DEBUG: Loading state dict from: {cfg.model_ckpt_path}")
    state_dict = torch.load(cfg.model_ckpt_path, map_location='cpu')
    print(f"DEBUG: State dict keys (first 5): {list(state_dict.keys())[:5]}")

    # Clean the state_dict keys to remove prefixes from torch.compile (`_orig_mod.`)
    # and the `.core` attribute used during training.
    cleaned_state_dict = {}
    prefix_to_strip = '_orig_mod.core.'
    print(f"DEBUG: Cleaning state dict with prefix: {prefix_to_strip}")
    for k, v in state_dict.items():
        if k.startswith(prefix_to_strip):
            new_k = k[len(prefix_to_strip):]
            cleaned_state_dict[new_k] = v
        else:
            # If a key doesn't have the prefix (e.g., from a different save method),
            # keep it as is, attempting a direct match.
            cleaned_state_dict[k] = v
    
    print(f"DEBUG: Cleaned state dict keys (first 5): {list(cleaned_state_dict.keys())[:5]}")
    print(f"DEBUG: Loading state dict into model...")
    model.load_state_dict(cleaned_state_dict)
    
    model = model.to(device).eval()
    print(f"DEBUG: Model moved to device: {device}")
    print(f"DEBUG: Model device after loading: {next(model.parameters()).device}")
    # model = model.to(device).bfloat16().eval()
    if cfg.compile:
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

    # --- 2. Load VAE Decoder ---
    print(f"Loading VAE decoder from checkpoint: {train_cfg.vae_ckpt_path}")
    print(f"DEBUG: VAE ID: {train_cfg.vae_id}")
    print(f"DEBUG: VAE config path: {train_cfg.vae_cfg_path}")
    decoder = get_decoder_only(
        train_cfg.vae_id,
        train_cfg.vae_cfg_path,
        train_cfg.vae_ckpt_path
    )
    # decoder = decoder.to(device).bfloat16().eval()
    decoder = decoder.to(device).eval()
    print(f"DEBUG: VAE decoder device: {next(decoder.parameters()).device}")
    if cfg.compile:
        print("Compiling VAE decoder...")
        decoder = torch.compile(decoder, mode='max-autotune', fullgraph=True)
    
    decode_fn = make_batched_decode_fn(decoder, vae_batch_size, temporal_vae=False)
    # The VAE for this model is not temporal, so set temporal_vae=False
    decode_fn = make_batched_decode_fn(decoder, train_cfg.vae_batch_size, temporal_vae=False)
    print(f"DEBUG: VAE batch size: {train_cfg.vae_batch_size}")
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
    
    print(f"DEBUG: Sample data ID: {train_cfg.sample_data_id}")
    print(f"DEBUG: Batch size: {cfg.batch_size}")
    print(f"DEBUG: Sample data kwargs: {train_cfg.sample_data_kwargs}")
    # Get the full dataset object to index into it
    sample_dataset = get_data_loader(
        sample_data_id,
        batch_size,
        **sample_data_kwargs
        train_cfg.sample_data_id,
        cfg.batch_size,
        **train_cfg.sample_data_kwargs
    ).dataset

    if starting_frame_index >= len(sample_dataset):
        raise ValueError(f"starting_frame_index {starting_frame_index} is out of bounds for dataset with length {len(sample_dataset)}")
    print(f"DEBUG: Dataset length: {len(sample_dataset)}")
    print(f"DEBUG: Starting frame index: {cfg.starting_frame_index}")
    
    if cfg.starting_frame_index >= len(sample_dataset):
        raise ValueError(f"starting_frame_index {cfg.starting_frame_index} is out of bounds for dataset with length {len(sample_dataset)}")

    data_dict = sample_dataset[starting_frame_index]
    initial_latents = data_dict['latents'].unsqueeze(0).to(device).bfloat16() / vae_scale
    
    if initial_context_length:
        initial_latents = initial_latents[:, :initial_context_length]
    
    print(f"Loaded context from data window index {starting_frame_index}")
    data_dict = sample_dataset[cfg.starting_frame_index]
    print(f"DEBUG: Data dict keys: {list(data_dict.keys())}")
    print(f"DEBUG: Raw latents shape: {data_dict['latents'].shape}")
    print(f"DEBUG: VAE scale: {train_cfg.vae_scale}")
    initial_latents = data_dict['latents'].unsqueeze(0).to(device).bfloat16() / train_cfg.vae_scale
    if cfg.initial_context_length:
        initial_latents = initial_latents[:, :cfg.initial_context_length]
    print(f"Loaded context from data window index {cfg.starting_frame_index}")
    
    print(f"Loading actions from '{actions_npy_path}'...")
    full_action_sequence = np.load(actions_npy_path)
    print(f"Loading actions from '{cfg.actions_npy_path}'...")
    full_action_sequence = np.load(cfg.actions_npy_path)
    print(f"DEBUG: Raw action sequence shape: {full_action_sequence.shape}")
    print(f"DEBUG: Raw action sequence dtype: {full_action_sequence.dtype}")
    
    # If the numpy array has multiple columns, assume the last one contains the action IDs
    if full_action_sequence.ndim > 1:
        print(f"Detected multi-column action file, using last column for action IDs.")
        print(f"DEBUG: Action sequence columns: {full_action_sequence.shape[1]}")
        full_action_sequence = full_action_sequence[:, -1]
    actions = torch.from_numpy(full_action_sequence).to(device).long().unsqueeze(0)
        print(f"DEBUG: After column selection shape: {full_action_sequence.shape}")
        
    actions = torch.from_numpy(full_action_sequence).to(device).long().unsqueeze(0) # Add batch dim
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
    print(f"DEBUG: Action tensor dtype: {actions.dtype}")
    print(f"DEBUG: Action tensor device: {actions.device}")

    print(f"Initial latents shape: {initial_latents.shape}")
    print(f"Full actions shape: {actions.shape}")
    print(f"DEBUG: Initial latents dtype: {initial_latents.dtype}")
    print(f"DEBUG: Initial latents device: {initial_latents.device}")

    # --- 4. Initialize Sampler ---
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
    print(f"DEBUG: Sampler ID: {train_cfg.sampler_id}")
    print(f"DEBUG: Original sampler kwargs: {train_cfg.sampler_kwargs}")
    sampler_kwargs = dict(train_cfg.sampler_kwargs) # Create a mutable copy
    sampler_kwargs['num_frames'] = cfg.num_frames
    print(f"DEBUG: Updated sampler kwargs: {sampler_kwargs}")
    sampler = get_sampler_cls(train_cfg.sampler_id)(**sampler_kwargs)
    print(f"Using sampler '{train_cfg.sampler_id}' to generate {cfg.num_frames} new frames.")
    print(f"DEBUG: Sampler type: {type(sampler)}")

    # --- 5. Run Inference ---
    print("Running inference...")
    print(f"DEBUG: About to run sampler with:")
    print(f"  - Model type: {type(model)}")
    print(f"  - Initial latents shape: {initial_latents.shape}")
    print(f"  - Actions shape: {actions.shape}")
    print(f"  - VAE scale: {train_cfg.vae_scale}")
    
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        video_out, _, out_actions = sampler(
            model,
        # The sampler expects the core model, which `model` already is.
        video_out, _, out_actions = sampler(
            model,
            initial_latents,
            actions,
            decode_fn=decode_fn,
            vae_scale=vae_scale
        )
    
    video_out = video_out.permute(0, 2, 1, 3, 4) 
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
    print(f"DEBUG: Video output shape: {video_out.shape if video_out is not None else 'None'}")
    print(f"DEBUG: Video output dtype: {video_out.dtype if video_out is not None else 'None'}")
    print(f"DEBUG: Video output device: {video_out.device if video_out is not None else 'None'}")
    print(f"DEBUG: Out actions shape: {out_actions.shape if out_actions is not None else 'None'}")

    # --- 6. Save Output ---
    if video_out is not None:
        print(f"Saving output video to '{cfg.output_path}'...")
        print(f"DEBUG: Video output before processing - shape: {video_out.shape}, dtype: {video_out.dtype}")
        print(f"DEBUG: Out actions before processing - shape: {out_actions.shape}, dtype: {out_actions.dtype}")
        
        # Process video frames with the same logic as to_wandb but without wandb dependency
        processed_frames = process_video_frames(video_out, out_actions, max_samples=4)
        print(f"Processed frames shape: {processed_frames.shape}")
        print(f"DEBUG: Processed frames dtype: {processed_frames.dtype}")
        print(f"DEBUG: Processed frames min/max values: {processed_frames.min()}/{processed_frames.max()}")
        
        if HAS_MOVIEPY:
            # Convert processed frames to the format expected by moviepy
            # processed_frames is [n,d,h,w] numpy array with values [0,255] uint8
            frame_list = []
            for i in range(processed_frames.shape[1]):
                # Convert from [d,h,w] to [h,w,d] and ensure it's in the right format
                frame = np.moveaxis(processed_frames[0, i], 0, -1) 
                # frame = np.moveaxis(processed_frames[i], 0, -1)  # [d,h,w] -> [h,w,d]
                frame_list.append(frame)
            
            print(f"DEBUG: Created frame list with {len(frame_list)} frames")
            print(f"DEBUG: First frame shape: {frame_list[0].shape if frame_list else 'No frames'}")
            
            # Create video using moviepy
            try:
                print("DEBUG: Creating ImageSequenceClip...")
                video_clip = ImageSequenceClip(frame_list, fps=30)
                print(f"DEBUG: Video clip duration: {video_clip.duration} seconds")
                print(f"DEBUG: Video clip size: {video_clip.size}")
                
                print("DEBUG: Writing video file...")
                video_clip.write_videofile(
                    cfg.output_path,
                    fps=30,
                    codec='libx264',
                    verbose=False,
                    logger=None
                )
                print("Video saved successfully.")
            except Exception as e:
                print(f"Error saving video with moviepy: {e}")
                print(f"DEBUG: Exception details: {type(e).__name__}: {str(e)}")
                # Fallback to numpy array
                print("Saving processed frames as numpy array...")
                np.save(cfg.output_path.replace('.mp4', '_frames.npy'), processed_frames)
                print(f"Frames saved as numpy array: {cfg.output_path.replace('.mp4', '_frames.npy')}")
        else:
            # Save as numpy array if moviepy is not available
            print("MoviePy not available. Saving processed frames as numpy array...")
            np.save(cfg.output_path.replace('.mp4', '_frames.npy'), processed_frames)
            print(f"Frames saved as numpy array: {cfg.output_path.replace('.mp4', '_frames.npy')}")
    else:
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
        "--config",
        type=str,
        default="configs/inference_config.yml",
        help="Path to the inference configuration YAML file."
        default="configs/inference_config.yml",
        help="Path to the inference configuration YAML file."
    )
    args = parser.parse_args()
    
    # Load the YAML configuration file
    inference_cfg = OmegaConf.load(args.config)
    run_inference(inference_cfg)
    
    # Load the YAML configuration file
    inference_cfg = OmegaConf.load(args.config)
    run_inference(inference_cfg)

