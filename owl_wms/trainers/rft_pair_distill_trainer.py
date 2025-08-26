import torch
import torch.nn.functional as F

from .world_trainer import WorldTrainer


class RFTPairDistillTrainer(WorldTrainer):
    def fwd_step(self, batch):
        return self.standard_loss_teacher(batch)

def standard_loss_teacher(self, batch):
    xs, t = batch["x_samples"], batch["times"]   # [B,N,K,C,H,W]
    B, N = xs.shape[:2]
    with torch.no_grad():
        ts = torch.randn(B, N, device=xs.device, dtype=xs.dtype).sigmoid()
        x0 = self.sample_xs_at_ts(xs, t, ts.new_zeros(B, N))  # t=0
        x1 = self.sample_xs_at_ts(xs, t, ts.new_ones (B, N))  # t=1
        x_t = torch.lerp(x0, x1, ts[..., None, None, None])
        v_target = x1 - x0
    with self.autocast_ctx:
        v_pred = self.core_fwd(x_t, ts)
    return F.mse_loss(v_pred.float(), v_target.float())

    def diffusion_forcing_distillation(self, batch):
        xs, t = batch["x_samples"], batch["times"]   # xs: [B,N,K,C,H,W]
        B, N = xs.shape[:2]

        with torch.no_grad():
            ts = torch.randn(B, N, device=xs.device, dtype=xs.dtype).sigmoid()
            x_t = self.sample_xs_at_ts(xs, t, ts)
            x0 = self.sample_xs_at_ts(xs, t, ts.new_zeros(B, N))
            x1 = self.sample_xs_at_ts(xs, t, ts.new_ones(B, N))
            v_target = x1 - x0

        with self.autocast_ctx:
            v_pred = self.core_fwd(x_t, ts)
        return F.mse_loss(v_pred.float(), v_target.float())

    @staticmethod
    def sample_xs_at_ts(xs, t, ts):
        """
        xs: [B,N,K,...], t: [K]/[B,K]/[B,N,K], ts: [B,N] in [0,1]
        returns x_t: [B,N,...]
        """
        B, N, K = xs.shape[:3]

        # normalize t -> [B,N,K]
        if t.dim() == 1:
            t = t.view(1, 1, K).expand(B, N, K)
        elif t.dim() == 2:
            t = t.view(B, 1, K).expand(B, N, K)
        else:
            assert t.shape == (B, N, K), f"Expected times [B,N,K], got {tuple(t.shape)}"

        t32 = t.to(device=xs.device, dtype=torch.float32)
        ts32 = ts.to(device=xs.device, dtype=torch.float32)

        # neighbors along K
        hi = torch.searchsorted(t32, ts32.unsqueeze(-1), right=True).clamp_(1, K - 1)
        lo = hi - 1

        # weights
        t_lo = torch.gather(t32, 2, lo).squeeze(2)
        t_hi = torch.gather(t32, 2, hi).squeeze(2)
        denom = (t_hi - t_lo).abs().clamp_min(torch.finfo(torch.float32).eps)
        w = ((ts32 - t_lo) / denom)

        # gather neighbors (match ndim)
        nd = xs.dim() - 3
        idx_lo = lo[(...,) + (None,) * nd]
        idx_hi = hi[(...,) + (None,) * nd]
        x_lo = torch.take_along_dim(xs, idx_lo, dim=2).squeeze(2)
        x_hi = torch.take_along_dim(xs, idx_hi, dim=2).squeeze(2)

        # interpolate
        w = w[(...,) + (None,) * nd].to(dtype=xs.dtype)
        return torch.lerp(x_lo, x_hi, w).contiguous()

    def standard_loss(self, batch):
        """
        x0: [B, N, C, H, W] clean latents (timestep 0.0)
        """
        x0 = batch["x_clean"]
        B, N = x0.size(0), x0.size(1)

        with torch.no_grad():
            ts = torch.randn(B, N, device=x0.device, dtype=x0.dtype).sigmoid()
            x1 = torch.randn_like(x0)  # gaussian input @ timestep 1.0
            x_t = x0 + (x1 - x0) * ts.view(B, N, 1, 1, 1)  # lerp to noise level @ ts
            v_target = x1 - x0

        with self.autocast_ctx:
            v_pred = self.core_fwd(x_t, ts)
        return F.mse_loss(v_pred.float(), v_target.float())

    def standard_loss_with_intermediate(self, batch):
        """
        Same as standard_loss, but uses the (student/teacher) intermediate latent (x_a, t_a)
        instead of fresh Gaussian noise (x1, 1.0).
        """
        x0, xa, ta = batch["x_clean"], batch["x_a"], batch["time_a"].float()
        B, N = x0.size(0), x0.size(1)

        with torch.no_grad():
            ts = torch.rand(B, N, device=x0.device)
            s = ts * ta
            x_t = x0 + (xa - x0) * ts.view(B, N, 1, 1, 1)
            v_target = xa - x0

        with self.autocast_ctx:
            v_pred = self.core_fwd(x_t, s)
        return F.mse_loss(v_pred.float(), v_target.float())

    @torch.compile
    def core_fwd(self, *args, **kwargs):
        core = self.get_module(ema=False).core
        return core(*args, **kwargs)
