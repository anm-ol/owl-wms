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

    def fwd_step(self, batch, train_step: int):
        x_a, t_a, x_b, t_b, x_clean, t_clean = batch  # t_clean unused

        # ----- Phase 1: ODE init (x_a, t_a) -> x_clean -----
        if train_step < self.train_cfg.finite_difference_step:
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
        u_e = u[..., None, None, None]                        # [B,F,1,1,1]

        x_u = x0 + u_e * v                                    # [B,F,C,H,W]
        t_u = u                                               # [B,F]

        pred_v = self.core_fwd(x_u, t_u)
        return F.mse_loss(pred_v, v)

    @torch.compile
    def core_fwd(self, *args, **kwargs):
        core = self.get_module(ema=False).core
        return core(*args, **kwargs)
