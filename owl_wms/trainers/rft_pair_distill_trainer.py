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
        x_a, time_a, x_b, time_b = batch

        scale = float(self.train_cfg.vae_scale)
        x_a = x_a / scale
        x_b = x_b / scale

        # Shared quantities from the pair
        denom = (time_b - time_a)  # [B, F]
        if torch.any(denom == 0):
            raise ValueError("t_b and t_a must differ (got zero Δt in batch).")

        # Broadcast [B,F] over [C,H,W]
        scale = 1.0 / denom.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,F,1,1,1]
        v = (x_b - x_a) * scale                                       # [B,F,C,H,W]
        x0 = (time_b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x_a
            - time_a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x_b) * scale  # [B,F,C,H,W]

        # Phase switch
        if train_step < self.train_cfg.finite_difference_step:
            # ----- ODE-pair regression: (x_a, time_a) -> x0 -----
            pred_x0 = self.core_fwd(x_a, time_a)
            diffusion_loss = F.mse_loss(pred_x0, x0)

        else:
            # ----- Flow-matching KD: (x_u, u) -> v -----
            # Choose u (fixed from cfg if provided, else random per step)

            # TODO: try random uniform
            # u = float(torch.rand((), device=x_a.device).item())
            # TODO: try midpoint between x_a and x_b
            # u = 0.5 * (t_a + t_b)
            # TODO: see how well strict midpoint below works
            # u = 0.5
            # random interpolated point
            u = time_b + torch.rand_like(time_a) * (time_a - time_b)

            x_u = x0 + u * v                             # [B,F,C,H,W]
            t_u = torch.full_like(time_a, u)             # [B,F]

            pred_v = self.core_fwd(x_u, t_u)
            diffusion_loss = F.mse_loss(pred_v, v)

        translation_loss = F.mse_loss(
            x_a,
            self.get_module(ema=False).core.translate_out(self.core.translate_in(x_a))
        )
        loss = diffusion_loss + translation_loss
        return loss, diffusion_loss, translation_loss


    @torch.compile
    def core_fwd(self, *args, **kwargs):
        core = self.get_module(ema=False).core
        return core(*args, **kwargs)
