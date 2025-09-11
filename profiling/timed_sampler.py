import torch
import time
from tqdm import tqdm
import gc
from collections import defaultdict

from owl_wms.sampling.t3_action_v2 import TekkenActionCachingV2
from owl_wms.models.tekken_rft_v2 import action_id_to_buttons
from owl_wms.sampling.schedulers import get_sd3_euler
from owl_wms.sampling.av_caching_v2 import get_deltas
from owl_wms.nn.kv_cache import KVCache

class SimpleProfiler:
    """A simple context manager to profile code blocks."""
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

    def report(self):
        print("\n--- Detailed Profiling Report ---")
        total_time = sum(self.timings.values())
        if total_time == 0:
            print("No timings recorded.")
            return
            
        print(f"{'Section':<25} | {'Total Time (ms)':>18} | {'Percentage':>12}")
        print("-" * 60)
        
        sorted_timings = sorted(self.timings.items(), key=lambda item: item[1], reverse=True)
        
        for key, value in sorted_timings:
            percentage = (value / total_time) * 100
            print(f"{key:<25} | {value:>18.2f} | {percentage:>11.2f}%")
        
        print("-" * 60)
        print(f"{'Total':<25} | {total_time:>18.2f} | {'100.00%':>12}")
        print("---------------------------------")


class ProfiledTekkenActionCachingV2(TekkenActionCachingV2):
    """
    An instrumented version of the TekkenActionCachingV2 sampler for detailed profiling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = SimpleProfiler()

    @torch.no_grad()
    def __call__(self, model, x, action_ids, compile_on_decode=False, decode_fn=None, 
                 means=None, stds=None, vae_scale=1.0):
        
        with self.profiler.time('Initialization'):
            batch_size, init_len = x.size(0), x.size(1)

            if self.custom_schedule is None:
                dt = get_sd3_euler(self.n_steps).to(device=x.device, dtype=x.dtype)
            else:
                dt = get_deltas(self.custom_schedule)

            kv_cache = KVCache(model.config)
            kv_cache.reset(batch_size)
            latents = [x.clone()]
            prev_x = x
            prev_actions = action_ids[:, :init_len]

        with self.profiler.time('Context Caching'):
            prev_x_noisy = self.zlerp(prev_x, self.noise_prev)
            prev_t = prev_x.new_full((batch_size, prev_x.size(1)), self.noise_prev)
            prev_actions_buttons = action_id_to_buttons(prev_actions)
            kv_cache.enable_cache_updates()
            _ = model(prev_x_noisy, prev_t, prev_actions_buttons, kv_cache=kv_cache)
            kv_cache.disable_cache_updates()

        def new_xt():
            return torch.randn_like(prev_x[:,:1]), prev_t.new_ones(batch_size, 1)

        num_frames = min(self.num_frames, action_ids.size(1) - init_len)
        if compile_on_decode:
            model = torch.compile(model)
        model.transformer.enable_decoding()
        
        for idx in tqdm(range(num_frames), desc="Profiling Tekken Sampler..."):
            with self.profiler.time('Frame Setup'):
                curr_x, curr_t = new_xt()
                current_frame_action = action_ids[:, init_len + idx:init_len + idx + 1]
                
            with self.profiler.time('Action Conversion (Denoise Loop)'):
                 current_frame_action_buttons = action_id_to_buttons(current_frame_action)

            for t_idx in range(self.n_steps):
                with self.profiler.time('Denoise Step (Model Forward)'):
                    pred_v = model(curr_x, curr_t, current_frame_action_buttons, kv_cache=kv_cache)
                
                with self.profiler.time('Denoise Step (Tensor Ops)'):
                    curr_x = curr_x - dt[t_idx] * pred_v   
                    curr_t = curr_t - dt[t_idx]

            with self.profiler.time('New Frame Caching'):
                latents.append(curr_x.clone())
                curr_x_noisy = self.zlerp(curr_x, self.noise_prev)
                curr_t_noisy = torch.ones_like(curr_t) * self.noise_prev
                current_frame_action_buttons_cache = action_id_to_buttons(current_frame_action)
                kv_cache.enable_cache_updates()
                _ = model(curr_x_noisy, curr_t_noisy, current_frame_action_buttons_cache, kv_cache=kv_cache)
                kv_cache.disable_cache_updates()
            
            with self.profiler.time('Window Truncation & GC'):
                if self.max_window is not None and len(latents) > self.max_window:
                    kv_cache.truncate(1, front=False)
                gc.collect()
                torch.cuda.empty_cache()

        with self.profiler.time('Finalization'):
            model.transformer.disable_decoding()
            generated_latents = torch.cat(latents, dim=1)
            # ... (rest of the finalization logic, which is fast) ...

        # After the call, you can access self.profiler.timings
        return None, None, None # Return dummy values for profiling
