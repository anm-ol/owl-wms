import torch
import torch.nn.functional as F


from .rft_trainer import RFTTrainer


class RFTPairDistillTrainer(RFTTrainer):
    def fwd_step(self, batch, train_step: int):
        if train_step < self.train_cfg.finite_difference_step:
            return self.ode_fwd(batch)
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
            v_pred = self.core_fwd(x_u, s_u)
        return F.mse_loss(v_pred.float(), v_hat.float())

    @torch.compile
    def core_fwd(self, *args, **kwargs):
        core = self.get_module(ema=False).core
        return core(*args, **kwargs)
