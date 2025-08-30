from ema_pytorch import EMA
from pathlib import Path
from tqdm import tqdm
import wandb
import gc

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer
from ..utils import freeze, Timer
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..muon import init_muon
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb_gif, to_wandb, to_wandb_pose
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn

class TekkenRFTTrainerV2(BaseTrainer):
    """
    A specialized trainer for the TekkenRFT model with periodic evaluation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = get_model_cls(self.model_cfg.model_id)(self.model_cfg)

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.total_step_counter = 0

        # VAE decoder is only needed on the main process for logging samples.
        mean_values = self.train_cfg.get('per_channel_mean', None)
        std_values = self.train_cfg.get('per_channel_std', None)

        if mean_values is not None and std_values is not None:
            # Get the number of channels from the model config
            C = self.model_cfg.channels
            
            # Convert lists to tensors and reshape for broadcasting
            # Assuming video tensor shape is (B, C, T, H, W) or (B, T, C, H, W)
            # The key is to align the C dimension. Let's assume the data loader
            # provides a tensor of shape (B, T, H, W, C), which is common.
            # If your shape is (B, C, T, H, W), use .view(1, C, 1, 1, 1) instead.
            self.per_channel_mean = torch.tensor(mean_values).view(1, 1, C, 1, 1)
            self.per_channel_std = torch.tensor(std_values).view(1, 1, C, 1, 1)
            if self.rank == 0:
                print("✓ Using per-channel mean and std for normalization.")
        else:
            self.per_channel_mean = None
            self.per_channel_std = None

        self.decoder = None
        if self.rank == 0:
            self.decoder = get_decoder_only(
                self.train_cfg.vae_id,
                self.train_cfg.vae_cfg_path,
                self.train_cfg.vae_ckpt_path
            )
            freeze(self.decoder)

    def save(self):
        save_dict = {
            'model': self.get_module().state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'steps': self.total_step_counter
        }
        super().save(save_dict)

    def load(self):
        if hasattr(self.train_cfg, 'resume_ckpt') and self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
            self.get_module().load_state_dict(save_dict['model'])
            self.ema.load_state_dict(save_dict['ema'])
            self.opt.load_state_dict(save_dict['opt'])
            self.total_step_counter = save_dict.get('steps', 0)

    def get_sample_data(self):
        """Create a fresh sample batch for evaluation"""
        sample_loader = get_loader(self.train_cfg.sample_data_id, self.train_cfg.n_samples, **self.train_cfg.sample_data_kwargs)
        return next(iter(sample_loader))

    def eval_step(self, sampler, decode_fn):
        """
        Runs a single evaluation step, generating a video sample and logging it.
        This function will only execute on the rank 0 process.
        """
        if self.rank != 0:
            return {}

        print("\nRunning evaluation step...")
        
        # Get fresh sample data each time to avoid StopIteration
        try:
            vid_for_sample, actions_for_sample, _ = self.get_sample_data()
        except Exception as e:
            print(f"Failed to get sample data: {e}")
            return {}
        

        if self.per_channel_mean is not None:
            # Broadcasting now works correctly
            print('Normalizing')
            print(f"Latent range before norm: min={vid_for_sample.min().item():.4f}, max={vid_for_sample.max().item():.4f}")
            # initial_latents = (vid_for_sample.cuda().bfloat16() - self.per_channel_mean) / self.per_channel_std
            initial_latents = (vid_for_sample.cuda().bfloat16())
            print(f"Latent range after norm: min={initial_latents.min().item():.4f}, max={initial_latents.max().item():.4f}")
        else:
            initial_latents = vid_for_sample.cuda().bfloat16() / self.train_cfg.vae_scale

        # Compute validation loss using EMA model
        action_ids_val = actions_for_sample.cuda()[:, :, -1].int()
        with torch.no_grad(), torch.amp.autocast('cuda', torch.bfloat16):
            ema_model = self.get_module(ema=True)
            val_loss = ema_model(initial_latents, action_ids=action_ids_val)
        
        num_repeats = (sampler.num_frames // actions_for_sample.shape[1]) + 2
        actions_for_sample = actions_for_sample.repeat(1, num_repeats, 1) # (b, t*repeat, 8)
        
        # Using your existing action conversion logic for compatibility.
        action_ids = actions_for_sample.cuda()[:, :, -1].int() # (b, t, 8) -> (b, t)

        with torch.no_grad(), torch.amp.autocast('cuda', torch.bfloat16):
            ema_core = self.get_module(ema=True).core if self.world_size > 1 else self.get_module(ema=True).core
            # print(f'this is the EMA core: {ema_core}')
            video_out, _, out_actions = sampler(
                ema_core,
                initial_latents,
                action_ids,
                decode_fn=decode_fn,
                means = self.per_channel_mean,
                stds = self.per_channel_std,
                vae_scale=self.train_cfg.vae_scale
            )
        video_out = video_out.permute(0, 2, 1, 3, 4)
        print(f'Generated video shape: {video_out.shape}, actions: {out_actions.shape}')
        c = video_out.shape[2]
        if c == 3:
            wandb_videos = to_wandb(video_out.cpu(), action_ids.cpu(), format='mp4', max_samples=self.train_cfg.n_samples, fps=10)
        else:
            rgb_videos, pose_videos = to_wandb_pose(video_out.cpu(), action_ids.cpu(), format='mp4', max_samples=self.train_cfg.n_samples, fps=10)

        del video_out, out_actions, initial_latents, actions_for_sample, action_ids, vid_for_sample
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Evaluation step finished.")
        if c == 3:
            return {"samples": wandb_videos, "val_loss": val_loss.item()}
        else:
            return {"rgb_samples": rgb_videos, "pose_samples": pose_videos, "val_loss": val_loss.item()}

    def train(self):
        torch.cuda.set_device(self.local_rank)

        self.model = self.model.cuda().train()
        
        if self.train_cfg.compile:
            print("Compiling the main model...")
            self.model = torch.compile(self.model)

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        
        decode_fn, sampler = None, None
        if self.rank == 0:
            self.decoder = self.decoder.cuda().eval().bfloat16()
            
            if self.train_cfg.compile:
                print("Compiling the VAE decoder...")
                self.decoder = torch.compile(self.decoder, mode="reduce-overhead", fullgraph=True)
            if self.train_cfg.data_kwargs.temporal_compression is None or self.train_cfg.data_kwargs.temporal_compression <= 1:
                temporal_vae = False
            else:
                temporal_vae = True
            decode_fn = make_batched_decode_fn(self.decoder, batch_size=self.train_cfg.vae_batch_size, temporal_vae=temporal_vae)
            sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        self.ema = EMA(self.model, beta=0.999, update_every=1)
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        accum_steps = max(1, self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size)
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        # get latents mean and std if there in the config
        # Check if mean and std are provided in the config
        if self.per_channel_mean is not None:
            self.per_channel_mean = self.per_channel_mean.cuda()
            self.per_channel_std = self.per_channel_std.cuda()

        self.load()

        metrics = LogHelper()
        if self.rank == 0:
            pass
            # wandb.watch(self.get_module(), log='all')

        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        print(f"Data loader created with {len(loader)} batches.")
        
        local_step = 0
        for epoch in range(self.train_cfg.epochs):
            if self.world_size > 1 and hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)

            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.train_cfg.epochs}", disable=self.rank != 0):
                batch_vid, batch_actions, batch_states = [t.cuda() for t in batch]
                
                # NOTE: This is your current action handling logic. For best results, consider
                # converting the full 8-bit vector to an integer ID.
                # print(f'Shape of batch_vid: {batch_vid.shape}, actions: {batch_actions.shape}, states: {batch_states.shape}')
                action_ids = batch_actions[:, :, -1].int() # (b, t, 8) -> (b, t)

                if self.per_channel_mean is not None:
                    # Broadcasting now works correctly
                    print('Normalizing')
                    print(f"Latent range before norm: min={batch_vid.min().item():.4f}, max={batch_vid.max().item():.4f}")
                    batch_vid = (batch_vid - self.per_channel_mean) / self.per_channel_std
                    print(f"Latent range after norm: min={batch_vid.min().item():.4f}, max={batch_vid.max().item():.4f}")
                    batch_vid = (batch_vid - self.per_channel_mean) / self.per_channel_std
                else:
                    batch_vid = batch_vid / self.train_cfg.vae_scale
                
                batch_vid = batch_vid.bfloat16()

                with ctx:
                    loss = self.model(batch_vid, action_ids=action_ids)
                    loss = loss / accum_steps
                
                loss.backward()
                metrics.log('diffusion_loss', loss.item() * accum_steps)
                
                local_step += 1

                if (local_step) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.ema.update()

                    self.total_step_counter += 1
                    
                    wandb_dict = metrics.pop()
                    if self.rank == 0:
                        wandb_dict['lr'] = self.opt.param_groups[0]['lr']

                    if self.total_step_counter % self.train_cfg.sample_interval == 0:
                        # Pass sampler and decode_fn instead of sample_loader
                        eval_wandb_dict = self.eval_step(sampler, decode_fn)
                        if self.rank == 0:
                            wandb_dict.update(eval_wandb_dict)
                    # Add a barrier here to make all processes wait for rank 0 to finish sampling
                    self.barrier()
                    
                    if self.rank == 0:
                        wandb.log(wandb_dict)

                    if self.total_step_counter % self.train_cfg.save_interval == 0 and self.rank == 0:
                        self.save()
                    
                    self.barrier()
                    
                    
if __name__ == '__main__':
    import sys
    import yaml
    import os
    from pathlib import Path
    from ..configs import Config
    from ..sampling import get_sampler_cls
    from ..data import get_loader
    from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
    
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Environment variables loaded from .env file")
    except ImportError:
        # If python-dotenv is not installed, try to load manually
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            print("✓ Environment variables loaded manually from .env file")
        else:
            print("⚠ No .env file found, WANDB functionality may not work")
    
    # Load config from yaml file
    config_path = Path("configs/tekken_dit_video.yml")
    print(f"Loading config from: {config_path}")
    
    try:
        # Use the proper Config.from_yaml method
        cfg = Config.from_yaml(config_path)
        print("✓ Config loaded successfully")
        print(f"Model: {cfg.model.model_id}")
        print(f"Sample size: {cfg.model.sample_size}")
        print(f"N frames: {cfg.model.n_frames}")
        
        # Create trainer instance (rank 0 only for testing)
        trainer = TekkenRFTTrainer(cfg.train, cfg.wandb, cfg.model, global_rank=0, local_rank=0, world_size=1)
        print("✓ Trainer created successfully")
        
        # Test model initialization
        print(f"✓ Model loaded: {type(trainer.model).__name__}")
        
        # Test VAE decoder setup
        if trainer.decoder is not None:
            print("✓ VAE decoder loaded successfully")
            decode_fn = make_batched_decode_fn(trainer.decoder, cfg.train.vae_batch_size)
            print("✓ Decode function created")
        else:
            print("✗ VAE decoder failed to load")
            sys.exit(1)
        
        # Test sampler creation
        try:
            sampler_cls = get_sampler_cls(cfg.train.sampler_id)
            if sampler_cls is not None:
                sampler = sampler_cls(**cfg.train.sampler_kwargs)
                print("✓ Sampler created successfully")
            else:
                print(f"✗ Sampler class not found: {cfg.train.sampler_id}")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Sampler creation failed: {e}")
            sys.exit(1)
        
        # Test data loader creation
        try:
            # Try to get sample data config, fallback to regular data config
            sample_data_id = getattr(cfg.train, 'sample_data_id', cfg.train.data_id)
            sample_data_kwargs = getattr(cfg.train, 'sample_data_kwargs', cfg.train.data_kwargs)
            
            sample_loader = get_loader(
                sample_data_id, 
                cfg.train.n_samples, 
                **sample_data_kwargs
            )
            if sample_loader is not None:
                print("✓ Sample data loader created successfully")
            else:
                print("✗ Data loader creation returned None")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Data loader creation failed: {e}")
            print("This might be due to missing data files in the specified directory")
            data_dir = cfg.train.data_kwargs.get('root_dir', 'not specified') if hasattr(cfg.train, 'data_kwargs') else 'not specified'
            print(f"Data directory: {data_dir}")
            sys.exit(1)
        
        # Test eval step
        print("\n=== Testing eval_step ===")
        try:
            # Initialize EMA for eval
            trainer.ema = EMA(trainer.model, beta=0.999, update_every=1)
            
            # Move model to GPU and set to eval mode for testing
            if torch.cuda.is_available():
                trainer.model = trainer.model.cuda().eval()
                trainer.decoder = trainer.decoder.cuda().eval().bfloat16()
                print("✓ Model and decoder moved to GPU")
            else:
                print("⚠ CUDA not available, running on CPU (slower)")
                trainer.model = trainer.model.eval()
                trainer.decoder = trainer.decoder.eval()
            
            # Test the eval step with new method signature
            eval_results = trainer.eval_step(sampler, decode_fn)
            
            if eval_results and "samples" in eval_results:
                print("✓ Eval step completed successfully!")
                print(f"✓ Generated samples returned")
            else:
                print("✗ Eval step returned empty results")
                
        except Exception as e:
            print(f"✗ Eval step failed: {e}")
            import traceback
            traceback.print_exc()
            
        print("\n=== Test Summary ===")
        print("Basic eval code test completed. Check above for any errors.")
        
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        print("Make sure you're running from the correct directory")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)