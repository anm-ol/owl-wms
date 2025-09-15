# inference/tekken_pipeline.py

import torch
import os
import glob
import random
import gc
from tqdm import tqdm
import numpy as np
import time

# Add project root to path to allow for module imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from owl_wms.configs import Config
from owl_wms.models import get_model_cls
from owl_wms.data import get_loader as get_data_loader
from owl_wms.nn.kv_cache import KVCache, StaticCache
from owl_wms.nn.rope import cast_rope_buffers_to_fp32
from owl_wms.utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from owl_wms.models.tekken_rft_v2 import action_id_to_buttons
from owl_wms import from_pretrained

def zlerp(x, alpha):
    return x * (1. - alpha) + alpha * torch.randn_like(x)

@torch.no_grad()
def to_bgr_uint8(frame, target_size=(1080, 1920)):
    """Converts a tensor to a BGR uint8 image."""
    frame = frame.flip(0)  # RGB to BGR
    frame = frame.permute(1, 2, 0)
    frame = (frame + 1) * 127.5
    frame = frame.clamp(0, 255).to(device='cpu', dtype=torch.uint8, memory_format=torch.contiguous_format, non_blocking=True)
    return frame

SAMPLING_STEPS = 4
WINDOW_SIZE = 60

class TekkenPipeline:
    """
    An interactive, real-time inference pipeline for Tekken world models.
    Based on CausvidPipeline but adapted for Tekken-specific data and inputs.
    """
    def __init__(self, 
                 cfg_path="configs/tekken_nopose_dmd.yml", 
                 ckpt_path="/mnt/data/laplace/owl-wms/checkpoints/tekken_nopose_dmd_L_ema/step_1500.pt", 
                 ground_truth=False,):
        
        print("🚀 Initializing Tekken Pipeline...")
        
        # Load configurations
        cfg = Config.from_yaml(cfg_path) 
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train
        self.ground_truth = ground_truth
        self.sampling_steps = self.train_cfg.sampler_kwargs.sampling_steps if hasattr(self.train_cfg, 'sampler_kwargs') and 'sampling_steps' in self.train_cfg.sampler_kwargs else SAMPLING_STEPS
        print(f"Using {self.sampling_steps} sampling steps.")
        # Load the core transformer model and the VAE frame decoder
        print("Loading models...")
        self.model, self.frame_decoder = from_pretrained(cfg_path, ckpt_path, True)
        self.model = self.model.core.cuda().bfloat16().eval()
        cast_rope_buffers_to_fp32(self.model)

        self.frame_decoder = self.frame_decoder.cuda().bfloat16().eval()

        # Store scaling factor (matching CausvidPipeline naming)
        self.frame_scale = self.train_cfg.vae_scale 
        self.image_scale = self.train_cfg.vae_scale
        
        # Alpha parameter for noising during cache building
        self.alpha = 0.25

        # Compile models for performance
        print("Compiling models...")
        self.model = torch.compile(self.model)  # Less aggressive compilation like CausvidPipeline
        self.frame_decoder = torch.compile(self.frame_decoder, mode='max-autotune', dynamic=False, fullgraph=True)
        
        self.device = 'cuda'
        
        # Initialize data buffers (using CausvidPipeline naming convention)
        self.history_buffer = None
        self.action_buffer = None  # This will hold our Tekken actions
        
        # Initial state storage for resets
        self._initial_history_buffer = None
        self._initial_action_buffer = None

        if self.ground_truth:
            self.future_action_buffer = None
            self._initial_future_action_buffer = None
            self.gt_step = 0

        # Initialize KV cache
        self.cache = None
        
        self.init_buffers()
        print("✅ Pipeline ready.")

    def _build_cache(self):
        """Build cache similar to CausvidPipeline but with Tekken actions."""
        print("Building KV cache with initial context...")
        
        # Disable decoding mode for cache building
        self.model.transformer.disable_decoding()
        batch_size = 1
        init_len = self.history_buffer.size(1)
        
        # Initialize KV cache (using StaticCache like CausvidPipeline)
        self.cache = StaticCache(self.model.config, max_length=init_len, batch_size=batch_size)
        
        # Noise the history buffer for caching (like CausvidPipeline)
        prev_x_noisy = zlerp(self.history_buffer, self.alpha)
        prev_t = self.history_buffer.new_full((batch_size, init_len), self.alpha)
        
        # Convert action IDs to button presses for the model
        button_presses = action_id_to_buttons(self.action_buffer)
        
        # Cache the context
        self.cache.enable_cache_updates()
        _ = self.model(
            prev_x_noisy,
            prev_t,
            button_presses,  # Tekken uses button presses instead of mouse+button
            kv_cache=self.cache
        )
        self.cache.disable_cache_updates()
        self.model.transformer.enable_decoding()

        # Garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    def init_buffers(self, starting_frame_index=0, future_action_path=None, window_size=WINDOW_SIZE):
        """Initialize data buffers using Tekken dataset."""
        print("Initializing data buffers...")
        
        # Use the validation data loader to get a starting window
        sample_dataset = get_data_loader(
            self.train_cfg.sample_data_id,
            1,  # Batch size of 1
            **self.train_cfg.sample_data_kwargs
        ).dataset

        if starting_frame_index >= len(sample_dataset):
            raise ValueError(f"starting_frame_index is out of bounds.")
        print(f"Using starting frame index: {starting_frame_index}")
        starting_frame_index += 200
        # Get initial context from the dataset
        data_dict = sample_dataset[starting_frame_index]
        
        if self.ground_truth:
            if not future_action_path:
                raise ValueError("`future_action_path` must be provided for ground truth mode.")
            
            # Load initial context
            initial_latents = data_dict['latents'].unsqueeze(0).to(self.device).bfloat16() / self.frame_scale
            initial_actions = data_dict['actions'][:, -1].unsqueeze(0).to(self.device).long()
            
            # Load future actions
            print(f"Loading future actions from {future_action_path} for ground truth mode...")
            full_action_sequence = np.load(future_action_path)
            if full_action_sequence.ndim > 1:
                full_action_sequence = full_action_sequence[:, -1]
            self.future_action_buffer = torch.from_numpy(full_action_sequence).to(self.device).long().unsqueeze(0)
            self.gt_step = 0
            
            self.history_buffer = initial_latents
            self.action_buffer = initial_actions
            
        else:
            # Standard initialization
            initial_latents = data_dict['latents'].unsqueeze(0).to(self.device).bfloat16() / self.frame_scale
            initial_actions = data_dict['actions'][:, -1].unsqueeze(0).to(self.device).long()
            
            self.history_buffer = initial_latents
            self.action_buffer = initial_actions

        # print(f"Loaded context from validation data index {starting_frame_index}.")
        # print(f"  - History Buffer Shape: {self.history_buffer.shape}")
        # print(f"  - Action Buffer Shape: {self.action_buffer.shape}")

        # Store initial state for resets
        self._initial_history_buffer = self.history_buffer.clone()
        self._initial_action_buffer = self.action_buffer.clone()
        if self.ground_truth:
            self._initial_future_action_buffer = self.future_action_buffer.clone()

        # Build the KV cache
        # print("model params dtype:", self.model.type)
        # print("history_buffer dtype:", self.history_buffer.dtype)
        # print("action_buffer dtype:", self.action_buffer.dtype)
        # print("future_action_buffer dtype:", self.future_action_buffer.dtype if self.ground_truth else "N/A")
        self._build_cache()

    def restart_from_buffer(self):
        """Restore buffers to their initial state."""
        print("🔄 Restarting pipeline from initial buffers.")
        self.history_buffer = self._initial_history_buffer.clone()
        self.action_buffer = self._initial_action_buffer.clone()
        
        if self.ground_truth:
            self.future_action_buffer = self._initial_future_action_buffer.clone()
            self.gt_step = 0

        self._build_cache()

    @torch.no_grad()
    def __call__(self, new_action_id: int = 0):
        """
        Generate next video frame based on action ID (following CausvidPipeline structure).
        
        Args:
            new_action_id (int): The discrete action ID for the next frame.

        Returns:
            A tuple containing the generated frame as a BGR uint8 tensor and the inference time in seconds.
        """
        if self.ground_truth:
            # Use ground truth actions instead of player inputs
            if self.gt_step >= self.future_action_buffer.size(1):
                print("⚠️ Ground truth action sequence exhausted. Using action ID 0.")
                current_action = torch.tensor([[0]], device=self.device, dtype=torch.long)
            else:
                current_action = self.future_action_buffer[:, self.gt_step:self.gt_step+1]
            self.gt_step += 1
        else:
            # Use player input
            current_action = torch.tensor([[new_action_id]], device=self.device, dtype=torch.long)
        
        # Convert action to button presses
        button_presses = action_id_to_buttons(current_action)  # [1,1,button_dim]
        
        # Initialize new frame as noise
        curr_x = torch.randn_like(self.history_buffer[:, :1])  # [1,1,c,h,w]
        curr_t = torch.ones(1, 1, device=curr_x.device, dtype=curr_x.dtype)  # [1,1]

        # Timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Sampling loop (matching CausvidPipeline's 2-step approach)
        if self.sampling_steps == 2:
            # First sampling step
            pred_v = self.model(
                curr_x,
                curr_t,
                button_presses,
                kv_cache=self.cache
            )
            
            curr_x = curr_x - 0.75 * pred_v
            curr_t = curr_t - 0.75

            # Second sampling step with cache update
            self.cache.enable_cache_updates()
            pred_v = self.model(
                curr_x,
                curr_t,
                button_presses,
                kv_cache=self.cache
            )
            self.cache.disable_cache_updates()

            new_frame = curr_x - 0.25 * pred_v
        else:
            # General sampling loop for other step counts
            dt = 1.0 / self.sampling_steps
            for step in range(self.sampling_steps):
                if step == self.sampling_steps - 1:
                    # Enable cache updates on the final step
                    self.cache.enable_cache_updates()
                
                pred_v = self.model(
                    curr_x,
                    curr_t,
                    button_presses,
                    kv_cache=self.cache
                )
                
                if step == self.sampling_steps - 1:
                    self.cache.disable_cache_updates()
                
                curr_x = curr_x - dt * pred_v
                curr_t = curr_t - dt

            new_frame = curr_x

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
        
        # Decode frame for display
        x_to_dec = new_frame[0] * self.image_scale
        frame = self.frame_decoder(x_to_dec).squeeze()  # [c,h,w]
        frame_image = to_bgr_uint8(frame)
        
        return frame_image, elapsed_time