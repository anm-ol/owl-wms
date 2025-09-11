import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm

# Add project root to Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from owl_wms.configs import Config
from owl_wms.models import get_model_cls
from owl_wms.sampling import get_sampler_cls
from owl_wms.data import get_loader as get_data_loader
from owl_wms.utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from owl_wms.utils.logging import to_wandb_video

def main(args):
    """
    Main function to run fast inference for the Tekken model.
    """
    # Load configuration from YAML file
    cfg = Config.from_yaml(args.config_path)
    model_cfg = cfg.model
    train_cfg = cfg.train

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print("Loading TekkenRFTv2 model...")
    model = get_model_cls(model_cfg.model_id)(model_cfg)
    model.load_state_dict(torch.load(args.model_ckpt_path, map_location='cpu'))
    model = model.to(device).bfloat16().eval()
    if args.compile:
        print("Compiling model for faster inference...")
        model = torch.compile(model, mode='max-autotune', fullgraph=True)
    print("Model loaded successfully.")

    # --- 2. Load VAE Decoder ---
    print("Loading VAE decoder...")
    decoder = get_decoder_only(
        train_cfg.vae_id,
        train_cfg.vae_cfg_path,
        train_cfg.vae_ckpt_path
    )
    decoder = decoder.to(device).bfloat16().eval()
    if args.compile:
        print("Compiling VAE decoder...")
        decoder = torch.compile(decoder, mode='max-autotune', fullgraph=True)
    decode_fn = make_batched_decode_fn(decoder, train_cfg.vae_batch_size, temporal_vae=False)
    print("VAE decoder loaded successfully.")

    # --- 3. Load Data ---
    print("Loading sample data for context...")
    sample_dataset = get_data_loader(
        train_cfg.sample_data_id,
        args.batch_size,
        **train_cfg.sample_data_kwargs
    ).dataset

    if args.starting_frame_index >= len(sample_dataset):
        raise ValueError(f"starting_frame_index {args.starting_frame_index} is out of bounds for dataset with length {len(sample_dataset)}")

    data_dict = sample_dataset[args.starting_frame_index]
    initial_latents = data_dict['latents'].unsqueeze(0).to(device).bfloat16() / train_cfg.vae_scale
    print(f"Loaded context from data window index {args.starting_frame_index}")
    
    print(f"Loading actions from '{args.actions_npy_path}'...")
    full_action_sequence = np.load(args.actions_npy_path)
    
    # If the numpy array has multiple columns, assume the last one contains the action IDs
    if full_action_sequence.ndim > 1:
        print(f"Detected multi-column action file, using last column for action IDs.")
        full_action_sequence = full_action_sequence[:, -1]
        
    actions = torch.from_numpy(full_action_sequence).to(device).long().unsqueeze(0) # Add batch dim
    print(f"Loaded action sequence of length {actions.shape[1]}")

    print(f"Initial latents shape: {initial_latents.shape}")
    print(f"Full actions shape: {actions.shape}")

    # --- 4. Initialize Sampler ---
    print("Initializing sampler...")
    sampler_kwargs = train_cfg.sampler_kwargs
    sampler_kwargs['num_frames'] = args.num_frames
    sampler = get_sampler_cls(train_cfg.sampler_id)(**sampler_kwargs)
    print(f"Using sampler '{train_cfg.sampler_id}' to generate {args.num_frames} new frames.")

    # --- 5. Run Inference ---
    print("Running inference...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        video_out, _, _ = sampler(
            model.core, # Sampler expects the core model
            initial_latents,
            actions,
            decode_fn=decode_fn,
            vae_scale=train_cfg.vae_scale
        )
    
    end_time.record()
    torch.cuda.synchronize()
    inference_time = start_time.elapsed_time(end_time) / 1000.0
    
    print(f"Inference completed in {inference_time:.2f} seconds.")

    # --- 6. Save Output ---
    if video_out is not None:
        print(f"Saving output video to '{args.output_path}'...")
        # to_wandb_video returns a wandb.Video object, we need to save its data.
        video_wandb = to_wandb_video(video_out, fps=30)
        with open(args.output_path, "wb") as f:
            f.write(video_wandb.data)
        print("Video saved successfully.")
    else:
        print("Inference did not produce a video output to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast inference script for the Tekken model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/tekken_nopose_large.yml",
        help="Path to the model configuration YAML file."
    )
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file)."
    )
    parser.add_argument(
        "--actions_npy_path",
        type=str,
        required=True,
        help="Path to the .npy file containing the full action sequence."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="tekken_inference_output.mp4",
        help="Path to save the generated video."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the data loader (for loading initial context)."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for model and VAE for faster inference."
    )
    parser.add_argument(
        "--starting_frame_index",
        type=int,
        default=0,
        help="The index of the starting data window from the validation set to use as context."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=160,
        help="Number of new frames to generate."
    )
    args = parser.parse_args()
    main(args)

