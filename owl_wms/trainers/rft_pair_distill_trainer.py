import torch
import torch.nn.functional as F


from .rft_trainer import RFTTrainer


class RFTPairDistillTrainer(RFTTrainer):
    """
    Unified pair-based distillation:
      batch = (x_a, time_a, x_b, time_b)

    - If train_step < finite_difference_step:
        ODE-pair regression (causvid P1): (x_a, time_a) -> x0
          x0 = (time_b * x_a - time_a * x_b) / (time_b - time_a)
    - Else:
        Flow-matching KD: (x_u, u) -> v
          v  = (x_b - x_a) / (time_b - time_a)
          x0 = (time_b * x_a - time_a * x_b) / (time_b - time_a)
          pick u in [0,1], x_u = x0 + u * v
    """

    def oldest_fwd_step(self, batch, train_step: int):
        x_a, t_a, x_b, t_b, x_clean, t_clean = batch  # t_clean unused

        # ----- Phase 1: ODE init (x_a, t_a) -> x_clean -----
        if train_step < self.train_cfg.finite_difference_step:
            with self.autocast_ctx:
                pred_x0 = self.core_fwd(x_a, t_a)
            return F.mse_loss(pred_x0, x_clean)

        # ----- Phase 2: Flow-matching KD (x_u, u) -> v -----
        denom = t_b - t_a  # [B,F]
        if torch.any(denom == 0):
            raise ValueError("t_b and t_a must differ (got zero Δt in batch).")

        inv = denom.reciprocal()[..., None, None, None]       # [B,F,1,1,1]
        t_a_e = t_a[..., None, None, None]                    # [B,F,1,1,1]
        t_b_e = t_b[..., None, None, None]                    # [B,F,1,1,1]

        v = (x_b - x_a) * inv                                # [B,F,C,H,W]
        x0 = (t_b_e * x_a - t_a_e * x_b) * inv                # [B,F,C,H,W]

        # TODO: try random uniform
        # u = float(torch.rand((), device=x_a.device).item())
        # TODO: try midpoint between x_a and x_b
        # u = 0.5 * (t_a + t_b)
        # TODO: see how well strict midpoint below works
        # u = 0.5
        # random interpolated point
        u = t_b + torch.rand_like(t_a) * (t_a - t_b)        # [B,F]

        x_u = x0 + u[..., None, None, None] * v                                    # [B,F,C,H,W]

        with self.autocast_ctx:
            pred_v = self.core_fwd(x_u, u)
        return F.mse_loss(pred_v, v)

    def old_fwd_step(self, batch, train_step: int):
        x_a, t_a, x_b, t_b, x_clean, t_clean = batch  # t_clean unused

        # ----- Phase 1: ODE init (x_a, t_a) -> x_clean -----
        if train_step < self.train_cfg.finite_difference_step:
            x_in, t_in = (x_a, t_a) if torch.rand(()) < 0.5 else (x_b, t_b)
            with self.autocast_ctx:
                pred_x0 = self.core_fwd(x_a, t_a)
            return F.mse_loss(pred_x0.float(), x_clean.float())

        # ----- Phase 2: Flow-matching KD (x_u, u) -> v -----
        denom = t_b - t_a  # [B,F]
        if torch.any(denom == 0):
            raise ValueError("t_b and t_a must differ (got zero Δt in batch).")

        inv = denom.reciprocal()[..., None, None, None]     # [B,F,1,1,1]
        t_a_e = t_a[..., None, None, None]                    # [B,F,1,1,1]

        v = ((x_b - x_a) * inv).detach()

        # Sample a single u per chunk (per video), then broadcast over frames
        u = t_b[:, :1] + torch.rand(t_a.size(0), 1, device=t_a.device, dtype=t_a.dtype) * (t_a[:, :1] - t_b[:, :1])  # [B,1]
        u_full = u.expand_as(t_a).contiguous()

        # Direct interpolation along the (a,b) segment at that u
        x_u = x_a + (u_full[..., None, None, None] - t_a_e) * (x_b - x_a) * inv     # [B,F,C,H,W]

        with self.autocast_ctx:
            pred_v = self.core_fwd(x_u, u_full)
        return F.mse_loss(pred_v.float(), v.float())

    def fwd_step(self, batch, train_step: int):
        if train_step < self.train_cfg.finite_difference_step:
            self.ode_fwd(batch)
        else:
            return self.flow_matching_fwd(batch)

    def ode_fwd(self, batch):
        x_a, t_a, _, _, x_clean, t_clean = batch
        with self.autocast_ctx:
            pred_x0 = self.core_fwd(x_a, t_a)
        return F.mse_loss(pred_x0.float(), x_clean.float())

    def flow_matching_fwd(self, batch, u_frac=None, noise_std=0.0):
        x_a, t_a, x_b, t_b, _, _ = batch

        # sample interpolated point
        if u_frac is None:
            u_frac = torch.rand_like(t_a)

        # inputs
        lam = u_frac.reshape(*u_frac.shape, *([1] * (x_a.dim() - u_frac.dim())))
        x_u = (1 - lam) * x_a + lam * x_b
        if noise_std:
            x_u = x_u + noise_std * torch.randn_like(x_u)

        s_u = (1 - u_frac) * t_a + u_frac * t_b

        # derivative
        dt = (t_b - t_a)  # shape [B]
        assert not torch.any(dt == 0)
        dt = dt.reshape(*dt.shape, *([1] * (x_a.dim() - dt.dim())))
        v_hat = (x_b - x_a) / dt

        # 3) model forward + loss
        with self.autocast_ctx:
            v_pred = self.model(x_u, s_u)
        return F.mse_loss(v_pred.float(), v_hat.float())

    @torch.compile
    def core_fwd(self, *args, **kwargs):
        core = self.get_module(ema=False).core
        return core(*args, **kwargs)
