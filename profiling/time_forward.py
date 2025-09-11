import torch
from torch import nn
import time
import sys
import os
from tqdm import tqdm
from collections import defaultdict
import einops as eo

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from owl_wms.models.tekken_rft_v2 import TekkenRFTCoreV2, action_id_to_buttons
from owl_wms.sampling.t3_action_v2 import TekkenActionCachingV2
from owl_wms.configs import Config
from owl_wms.nn.kv_cache import KVCache

class SimpleProfiler:
    """A simple context manager to profile code blocks and aggregate timings."""
    def __init__(self):
        self.timings = defaultdict(float)
        self.start_time = None
        self.current_key = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = (time.perf_counter() - self.start_time) * 1000  # ms
        if self.current_key:
            self.timings[self.current_key] += elapsed_time

    def time(self, key):
        self.current_key = key
        return self
    
    def add_timings(self, other_profiler):
        for key, value in other_profiler.timings.items():
            self.timings[key] += value

    def report(self):
        print("\n--- Detailed Forward Pass Profiling Report ---")
        total_time = sum(self.timings.values())
        if total_time == 0:
            print("No timings recorded.")
            return

        print(f"{'Operation':<30} | {'Total Time (ms)':>18} | {'Percentage':>12}")
        print("-" * 65)
        
        sorted_timings = sorted(self.timings.items(), key=lambda item: item[1], reverse=True)
        
        for key, value in sorted_timings:
            percentage = (value / total_time) * 100
            print(f"{key:<30} | {value:>18.2f} | {percentage:>11.2f}%")
        
        print("-" * 65)
        print(f"{'Total':<30} | {total_time:>18.2f} | {'100.00%':>12}")
        print("----------------------------------------------")

class ProfiledTekkenRFTCoreV2(TekkenRFTCoreV2):
    """An instrumented version of the model to profile the forward pass."""
    def __init__(self, config):
        super().__init__(config)
        self.profiler = SimpleProfiler()

    def forward(self, x, t, button_presses, has_controls=None, kv_cache=None):
        b, n, c, h, w = x.shape
        
        with self.profiler.time('t_embed'):
            t_cond = self.t_embed(t)
        
        with self.profiler.time('action_embed'):
            action_tokens = self.action_embed(button_presses)
            action_emb = action_tokens.mean(dim=2)
        
        with self.profiler.time('Conditioning Logic'):
             cond_emb = t_cond + action_emb

        with self.profiler.time('Reshape & Proj In'):
            x_tokens = eo.rearrange(x, 'b t c h w -> b t (h w) c')
            x_tokens = self.proj_in(x_tokens)
            
        with self.profiler.time('Transformer Input Prep'):
            b_t, t_t, s, d = x_tokens.shape
            transformer_input = x_tokens.view(b_t, t_t * s, d)
            cond = cond_emb.unsqueeze(2).expand(b_t, t_t, s, d).contiguous().view(b_t, t_t * s, d)

        with self.profiler.time('Transformer Backbone'):
            processed_tokens = self.transformer(transformer_input, cond, kv_cache=kv_cache)

        with self.profiler.time('Output Reshaping'):
            processed_tokens = processed_tokens.view(b_t, t_t, s, d)
            processed_video_tokens = processed_tokens.reshape(b_t, t_t * s, d)
            video_cond = t_cond.unsqueeze(2).expand(b_t, t_t, s, d).contiguous().view(b_t, t_t * s, d)
        
        with self.profiler.time('proj_out'):
            output_latents = self.proj_out(processed_video_tokens, video_cond)
        
        with self.profiler.time('Final Rearrange'):
            output = eo.rearrange(output_latents, 'b (t h w) c -> b t c h w', t=n, h=h, w=w)
            
        return output

def run_profiler():
    """Main function to set up and run the detailed profiler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = Config.from_yaml("configs/tekken_dmd.yml")
    model_config = config.model
    train_config = config.train
    
    # Use the profiled model
    model = ProfiledTekkenRFTCoreV2(model_config)
    model.to(device).eval()
    
    # Create dummy data
    context_length = 16
    num_frames_to_gen = 60
    total_sequence_length = context_length + num_frames_to_gen
    batch_size = 1
    
    initial_latents = torch.randn(batch_size, context_length, 128, 14, 23, device=device, dtype=torch.bfloat16)
    action_ids = torch.randint(0, 256, (batch_size, total_sequence_length), device=device)

    # Use the original sampler, but it will call our profiled model
    sampler_args = train_config.sampler_kwargs
    sampler_args['num_frames'] = num_frames_to_gen
    sampler = TekkenActionCachingV2(**sampler_args)

    print(f"Running detailed profiling for {num_frames_to_gen} frames...")
    
    # The sampler will internally call our profiled model's forward pass
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        sampler(model, initial_latents, action_ids)
    
    # Report the aggregated timings from the model's internal profiler
    model.profiler.report()

if __name__ == "__main__":
    run_profiler()
