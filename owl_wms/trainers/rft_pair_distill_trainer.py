import torch
import torch.nn.functional as F

from .world_trainer import WorldTrainer


class RFTPairDistillTrainer(WorldTrainer):
    def fwd_step(self, batch):
        return self.rectified_flow_teacher(batch)
        """
        if self.total_step_counter % 8 == 0:
            # Flow matching loss
            return self.flow_matching(batch)
        else:
            # Standard loss
            with self.autocast_ctx:
                return self.model(batch["x_clean"])
        """

    def rectified_flow_teacher(self, batch, u_frac: float | None = None, noise_std: float = 0.0):
        """
        Rectified-Flow (displacement) loss along the teacher segment (x_a → x_b).
        Uses a point on the straight path at s_u and targets (x_b - x_a).
        """
        x_a, t_a = batch["x_a"], batch["time_a"]      # [B,F,C,H,W], [B,F]
        x_b, t_b = batch["x_b"], batch["time_b"]      # [B,F,C,H,W], [B,F]

        # λ ∈ [0,1]
        u = torch.rand_like(t_a) if u_frac is None else torch.full_like(t_a, float(u_frac))

        # Interpolate state and time (keep times in fp32 into the embed)
        lam = u[..., None, None, None]                # [B,F,1,1,1]
        x_u = x_a + (x_b - x_a) * lam
        if noise_std:
            x_u = x_u + noise_std * torch.randn_like(x_u)
        s_u = (t_a + (t_b - t_a) * u).float()         # [B,F]

        # Displacement target and MSE
        v_hat = x_b - x_a                              # [B,F,C,H,W]
        with self.autocast_ctx:
            v_pred = self.core_fwd(x_u, s_u)          # core(x:[B,F,C,H,W], t:[B,F])
        return F.mse_loss(v_pred.float(), v_hat.float())

    def flow_matching(self, batch, u_frac=None, noise_std=0.0):
        x_a, t_a, x_b, t_b = batch["x_a"], batch["time_a"], batch["x_b"], batch["time_b"]

        # 1) λ ~ U(0,1) unless explicitly fixed
        if u_frac is None:
            u_frac = torch.rand_like(t_a)           # [B,F]
        else:
            u_frac = torch.full_like(t_a, float(u_frac))

        # 2) Interpolate state and time
        lam = u_frac.view(*u_frac.shape, *([1] * (x_a.dim() - u_frac.dim())))  # -> [B,F,1,1,1]
        x_u = (1 - lam) * x_a + lam * x_b
        if noise_std:
            x_u = x_u + noise_std * torch.randn_like(x_u)

        s_u = ((1 - u_frac) * t_a + u_frac * t_b).to(dtype=x_u.dtype)  # cast time to match latent dtype

        # 3) Constant-velocity target (use one dt per sample)
        dt = (t_b[:, 0] - t_a[:, 0])                                   # [B], note t_b < t_a
        dt = torch.where(dt.abs() < 1e-4, dt.sign() * 1e-4, dt)        # avoid tiny gaps
        dt = dt.view(-1, 1, 1, 1, 1)                                   # -> [B,1,1,1,1]
        v_hat = (x_b - x_a) / dt                                       # [B,F,C,H,W]

        # 4) Predict v(x_u, s_u) and plain MSE
        with self.autocast_ctx:
            v_pred = self.core_fwd(x_u, s_u)                           # core(x:[B,F,C,H,W], t:[B,F])
        return F.mse_loss(v_pred.float(), v_hat.float())

    @torch.compile
    def core_fwd(self, *args, **kwargs):
        core = self.get_module(ema=False).core
        return core(*args, **kwargs)
