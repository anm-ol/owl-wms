import argparse
import os
import sys
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as TF
from moviepy.editor import VideoFileClip, ImageSequenceClip

from owl_wms.configs import Config as WmsConfig

# Make sure we can import the VAE code
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(REPO_ROOT, "owl-vaes"))

from owl_vaes.configs import Config as OwlVaeConfig  # type: ignore
from owl_vaes.models import get_model_cls as get_owl_model_cls  # type: ignore
from owl_vaes.utils import versatile_load  # type: ignore


def load_experiment_config(path: str) -> WmsConfig:
    """Load the owl-wms experiment YAML (e.g. combat_v0_t3vae_s.yml)."""
    return WmsConfig.from_yaml(path)


def load_vae_from_train_cfg(train_cfg, device: torch.device) -> nn.Module:
    """Instantiate and load the VAE specified by train.vae_cfg_path / train.vae_ckpt_path."""
    if train_cfg.vae_cfg_path is None or train_cfg.vae_ckpt_path is None:
        raise ValueError("Config does not specify vae_cfg_path and vae_ckpt_path in train section.")

    vae_cfg_path = os.path.join(REPO_ROOT, train_cfg.vae_cfg_path)
    vae_ckpt_path = train_cfg.vae_ckpt_path

    # If ckpt path is relative, make it repo-relative too
    if not os.path.isabs(vae_ckpt_path):
        vae_ckpt_path = os.path.join(REPO_ROOT, vae_ckpt_path)

    if not os.path.exists(vae_cfg_path):
        raise FileNotFoundError(f"VAE config not found: {vae_cfg_path}")
    if not os.path.exists(vae_ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")

    owl_cfg = OwlVaeConfig.from_yaml(vae_cfg_path).model
    vae_cls = get_owl_model_cls(owl_cfg.model_id)
    vae: nn.Module = vae_cls(owl_cfg)

    state = versatile_load(vae_ckpt_path)
    vae.load_state_dict(state, strict=True)

    # Move to device and dtype similar to training scripts
    if device.type == "cuda":
        vae = vae.bfloat16().to(device)
    else:
        vae = vae.to(device)
    vae.eval()
    return vae


def read_video_frames(path: str, max_frames: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Read an MP4 into a numpy array [T, H, W, 3] uint8 and return fps."""
    clip = VideoFileClip(path)
    fps = float(clip.fps)

    frames: List[np.ndarray] = []
    for i, frame in enumerate(clip.iter_frames()):
        frames.append(frame)
        if max_frames is not None and i + 1 >= max_frames:
            break

    clip.close()

    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")

    video = np.stack(frames, axis=0).astype(np.uint8)  # [T, H, W, 3]
    return video, fps


@torch.no_grad()
def reconstruct_video(
    vae: nn.Module,
    video: np.ndarray,
    sample_size: Tuple[int, int],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run VAE reconstruction on video frames.

    Args:
        vae: loaded VAE model with .encoder and .decoder
        video: [T, H, W, 3] uint8
        sample_size: (H, W) expected by VAE
        device: torch device
        batch_size: frames per batch

    Returns:
        orig_resized: [T, H, W, 3] uint8 (resized to VAE resolution)
        recon:        [T, H, W, 3] uint8
    """
    T, H, W, C = video.shape
    target_h, target_w = sample_size

    # Convert to tensor [T, 3, H, W]
    frames = torch.from_numpy(video).permute(0, 3, 1, 2)  # uint8

    # Resize to VAE resolution
    frames = TF.resize(frames, size=[target_h, target_w], antialias=True)
    orig_resized = frames.clone()  # keep copy for side-by-side

    frames = frames.to(device=device, dtype=torch.float32)
    frames = (frames / 255.0) * 2.0 - 1.0  # [T, 3, H, W] in [-1, 1]

    recons: List[np.ndarray] = []

    use_autocast = device.type == "cuda"

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch = frames[start:end]  # [b,3,H,W]

        if use_autocast:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = vae.encoder(batch)
                # Some VAEs may return (mu, logvar) during training; in eval DCAE returns mu directly.
                if isinstance(z, tuple):
                    z = z[0]
                rec = vae.decoder(z)
        else:
            z = vae.encoder(batch)
            if isinstance(z, tuple):
                z = z[0]
            rec = vae.decoder(z)

        rec = rec.detach().to("cpu", dtype=torch.float32)
        rec = ((rec + 1.0) / 2.0).clamp(0.0, 1.0)
        rec = (rec * 255.0).round().to(torch.uint8)  # [b,3,H,W]
        rec_np = rec.permute(0, 2, 3, 1).numpy()      # [b,H,W,3]
        recons.append(rec_np)

    recon_video = np.concatenate(recons, axis=0)

    # Orig resized back to [T,H,W,3] uint8
    orig_resized_np = orig_resized.to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return orig_resized_np, recon_video


def make_side_by_side(orig: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Stack original and reconstructed frames horizontally: [T, H, 2W, 3]."""
    assert orig.shape == recon.shape, f"Shape mismatch: orig {orig.shape}, recon {recon.shape}"
    return np.concatenate([orig, recon], axis=2)


def write_video(frames: np.ndarray, fps: float, out_path: str) -> None:
    """Write frames [T, H, W, 3] uint8 as an MP4 file."""
    clip = ImageSequenceClip(list(frames), fps=fps)
    # No audio for now
    clip.write_videofile(out_path, codec="libx264", audio=False)
    clip.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test VAE reconstructions on an MP4 and save side-by-side video.")
    parser.add_argument("--config", required=True, help="Path to owl-wms YAML config (e.g. configs/combat/combat_v0_t3vae_s.yml)")
    parser.add_argument("--video", required=True, help="Path to input MP4 video")
    parser.add_argument("--output", required=True, help="Path to output MP4 video")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max number of frames to process")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for VAE inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (e.g. 'cuda', 'cuda:0', or 'cpu')")

    args = parser.parse_args()

    device = torch.device(args.device)

    # 1) Load experiment config and VAE
    exp_cfg = load_experiment_config(args.config)
    train_cfg = exp_cfg.train

    vae = load_vae_from_train_cfg(train_cfg, device=device)

    # Infer sample_size (H, W) from VAE config stored on the model
    if not hasattr(vae, "config") or not hasattr(vae.config, "sample_size"):
        raise AttributeError("Loaded VAE does not expose config.sample_size; cannot infer input resolution.")
    sample_size = tuple(vae.config.sample_size)  # (H, W)

    # 2) Load input video
    video_np, fps = read_video_frames(args.video, max_frames=args.max_frames)

    # 3) Run reconstruction
    orig_resized, recon = reconstruct_video(
        vae=vae,
        video=video_np,
        sample_size=sample_size,
        device=device,
        batch_size=args.batch_size,
    )

    # 4) Make side-by-side comparison and write to disk
    side_by_side = make_side_by_side(orig_resized, recon)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_video(side_by_side, fps=fps, out_path=args.output)


if __name__ == "__main__":
    main()
