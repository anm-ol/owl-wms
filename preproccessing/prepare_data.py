import os
import glob
from tqdm import tqdm
import torch
import numpy as np
import time
from encoder import load_vae, load_encoder
import subprocess


def save_video_ffmpeg(video_np, output_path, fps=30):
    """
    Save video frames using ffmpeg.

    Args:
        video_np: numpy array of shape [t, h, w, c] with values 0-255
        output_path: path to save the video
        fps: frames per second
    """
    num_frames, height, width, channels = video_np.shape

    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',  # The input comes from a pipe
        '-an',  # No audio
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
            process.stdin.write(video_np.tobytes())
            process.stdin.close()

            # Wait for process to complete
            return_code = process.wait()

            if return_code == 0:
                print(f"Video saved successfully to {output_path}")
            else:
                stderr_output = process.stderr.read().decode('utf-8')
                print(f"FFmpeg failed with return code {return_code}")
                print(f"FFmpeg stderr: {stderr_output}")

    except Exception as e:
        print(f"Error running ffmpeg: {e}")


def test_vae(vae, data_path, enable_tiling=False, encode_separately=False, device='cuda'):
    start_time = time.time()

    print("Loading frames...")
    load_start = time.time()
    frames = get_frames_from_npz(data_path)
    frames = frames.to(device)
    load_time = time.time() - load_start
    print(f"Frame loading time: {load_time:.2f}s")

    length = frames.shape[0]
    # input should have n * 4 + 1 frames
    n_frames_to_keep = length - (length - 1) % 4
    frames = frames[:n_frames_to_keep]
    frames = frames.to(torch.bfloat16)
    frames = frames.permute(1, 0, 2, 3)  # [t, c, h, w] -> [c, t, h, w]
    frames = frames.unsqueeze(0)  # Add batch dimension
    print(f"Input shape: {frames.shape}")  # [1, c, t, h, w]

    frames = (frames / 255.0) * 2 - 1

    # Enable VAE tiling if requested
    if enable_tiling:
        print("Enabling VAE tiling...")
        vae.enable_tiling()
    else:
        print("VAE tiling disabled")

    print("Running VAE inference...")
    inference_start = time.time()
    vae.eval()
    with torch.no_grad():
        if encode_separately:
            encoding_start = time.time()
            clean_latent_dist = vae.encode(frames).latent_dist
            encoding_time = time.time() - encoding_start
            print(f"Encoding time: {encoding_time:.2f}s")

            z0 = clean_latent_dist.sample()

            # 2. Define a small noise level 't'
            # The paper trained the decoder in the [0, 0.2] range. 0.05 is a good starting point.
            t_val = 0.0
            t = torch.tensor([t_val] * z0.shape[0], device=device, dtype=z0.dtype)

            # 3. Create noise and generate the noisy latent, zt
            noise = torch.randn_like(z0)
            zt = (1 - t_val) * z0 + t_val * noise

            # 4. Decode from the *noisy* latent, providing the timestep 't'
            decoding_start = time.time()
            decoded_output = vae.decode(zt, temb=t)
            decoded = decoded_output.sample
            decoding_time = time.time() - decoding_start
            print(f"Decoding time: {decoding_time:.2f}s")
        else:
            # Directly encode and decode in one step
            decoded = vae(frames).sample
            encoding_time = decoding_time = 0.0

    # Disable tiling after inference
    if enable_tiling:
        vae.disable_tiling()

    inference_time = time.time() - inference_start
    print(f"VAE inference time: {inference_time:.2f}s")

    print("Saving videos...")
    save_start = time.time()

    # Save reconstructed video
    reconstructed_tensor = decoded.squeeze(0).permute(1, 2, 3, 0).cpu()
    reconstructed_np = ((reconstructed_tensor + 1) / 2.0 * 255.0).to(torch.uint8).numpy()
    reconstructed_output_path = "reconstructed_video.mp4"
    save_video_ffmpeg(reconstructed_np, reconstructed_output_path, fps=24)

    save_time = time.time() - save_start
    print(f"Video saving time: {save_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\n=== Timing Summary ===")
    print(f"Frame loading: {load_time:.2f}s")
    print(f"VAE inference: {inference_time:.2f}s")
    print(f"  - Encoding: {encoding_time:.2f}s")
    print(f"  - Decoding: {decoding_time:.2f}s")
    print(f"Video saving: {save_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tiling enabled: {enable_tiling}")
    print(f"Reconstructed video saved to: {reconstructed_output_path}")


def process_data(data_dir, output_dir, vae_ckpt_dir, enable_tiling=False, device='cuda'):
    """
    Processes raw Tekken round data from .npz files, encodes video frames into latents
    using a VAE, and saves the latents, actions, and states in a nested directory structure.
    """
    print("--- Starting Data Processing ---")

    # 1. Load VAE
    vae = load_vae(vae_ckpt_dir, device=device)
    if enable_tiling:
        print("Enabling VAE tiling...")
        vae.enable_tiling()

    # 2. Find all .npz files in the data directory
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {data_dir}. Exiting.")
        return

    print(f"Found {len(npz_files)} rounds to process.")

    # 3. Process each file
    for npz_path in tqdm(npz_files, desc="Processing Rounds"):
        round_name = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"\nProcessing {round_name}...")

        # Create output directories for the current round
        round_output_dir = os.path.join(output_dir, round_name)
        latents_dir = os.path.join(round_output_dir, "latents")
        actions_dir = os.path.join(round_output_dir, "actions")
        states_dir = os.path.join(round_output_dir, "states")

        os.makedirs(latents_dir, exist_ok=True)
        os.makedirs(actions_dir, exist_ok=True)
        os.makedirs(states_dir, exist_ok=True)

        # Load data from .npz file
        data = np.load(npz_path)
        mask = data['attention_mask']
        end_idx = int(np.where(mask == 1)[0][-1])

        images = torch.from_numpy(data['images'][:end_idx]).float()
        video_len = images.shape[0]
        n_frames_to_keep = video_len - (video_len - 1) % 4
        images = images[:n_frames_to_keep]
        actions = data['actions'][:end_idx]
        states = data['states'][:end_idx]
        data.close()

        with torch.no_grad():
            images = images.to(device)
            images = (images / 255.0) * 2 - 1
            images = images.to(vae.dtype)  # Ensure correct dtype for VAE
            images = images.permute(1, 0, 2, 3)  # [t, c, h, w] -> [c, t, h, w]
            images = images.unsqueeze(0)  # Add batch dimension
            latent_dist = vae.encode(images).latent_dist
            latent_sample = latent_dist.sample()
            all_latents = latent_sample.to(torch.float16).cpu().numpy()

        final_latents = all_latents.squeeze(0)

        # Save the processed data
        np.save(os.path.join(latents_dir, f"{round_name}_latents.npy"), final_latents)
        np.save(os.path.join(actions_dir, f"{round_name}_actions.npy"), actions)
        np.save(os.path.join(states_dir, f"{round_name}_states.npy"), states)

        print(f"Finished processing and saved data for {round_name}.")

    if enable_tiling:
        vae.disable_tiling()

    print("\n--- Data Processing Complete ---")


def get_frames_from_npz(file_path):
    """
    Extracts frames from a .npz file and returns them as a tensor.
    """
    data = np.load(file_path)
    mask = data['attention_mask']
    idx = int(np.where(mask == 1)[0][-1])
    images = data['images'][:idx]
    video_tensor = torch.from_numpy(images).float()
    data.close()
    return video_tensor  # [t, c, h, w]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wan_path = "checkpoints/Wan2.1/vae"
    ltx_path = "checkpoints/LTXV/vae"

    process_data(
        data_dir="t3_data/",
        output_dir="cached_data_ltx",
        vae_ckpt_dir=ltx_path,
        enable_tiling=True,
        device='cuda'
    )
