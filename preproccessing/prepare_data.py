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
    
    len = frames.shape[0]
    # input should have n * 4 + 1 frames
    n_frames_to_keep = len - (len - 1) % 4
    frames = frames[:n_frames_to_keep]
    # frames = frames[:17]
    frames = frames.to(torch.bfloat16)
    frames = frames.permute(1, 0, 2, 3)  # [t, c, h, w] -> [c, t, h, w]
    frames = frames.unsqueeze(0)  # Add batch dimension
    print(f"Input shape: {frames.shape}")  # [1, c, t, h, w]
    
    frames = (frames/255.0) * 2 - 1
    
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
    
    # Save original video
    # original_tensor = frames.squeeze(0).permute(1, 2, 3, 0).cpu()
    # original_np = ((original_tensor + 1) / 2.0 * 255.0).to(torch.uint8).numpy()
    # original_output_path = "original_video.mp4"
    # save_video_ffmpeg(original_np, original_output_path, fps=24)
    
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
    # print(f"Original video saved to: {original_output_path}")
    print(f"Reconstructed video saved to: {reconstructed_output_path}")

def get_frames_from_npz(file_path):
    """
    Extracts frames from a .npz file and returns them as a tensor.
    
    Args:
        file_path: Path to the .npz file containing video frames.
        
    Returns:
        Tensor of shape [b, c, h, w] where b is the number of frames,
        c is the number of channels, h is height, and w is width.
    """
    data = np.load(file_path)
    mask = data['attention_mask']
    idx = int(np.where(mask == 1)[0][-1])
    images = data['images'][:idx]
    video_tensor = torch.from_numpy(images).float()
    data.close()  # Free memory
    return video_tensor  # [t,c,h,w]
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wan_path = "checkpoints/Wan2.1/vae"
    ltx_path = "checkpoints/LTXV/vae"
    vae = load_vae(wan_path, dtype=torch.bfloat16, device=device)
    
    # Test with tiling enabled
    test_vae(vae, "t3_data/tekken_dataset_npz/round_001.npz", enable_tiling=False, encode_separately=True)
