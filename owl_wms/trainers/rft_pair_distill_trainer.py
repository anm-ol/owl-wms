import torch
import torch.nn.functional as F
import random

from .craft_trainer import CraftTrainer


class RFTPairDistillTrainer(CraftTrainer):
    def fwd_step(self, batch, train_step: int):
        if train_step % 8 == 0:
            return self.flow_matching_fwd2(batch)
        else:
            with self.autocast_ctx:
                return self.model(*batch)

        # return self.ayf_emd(batch)
        """
        if train_step < self.train_cfg.finite_difference_step:
            return self.ode_fwd(batch)
        else:
            return self.flow_matching_fwd(batch)
        """

    def ode_fwd(self, batch):
        x_a, t_a, _, _, x_clean, t_clean = batch
        with self.autocast_ctx:
            pred_x0 = self.core_fwd(x_a, t_a)
        return F.mse_loss(pred_x0.float(), x_clean.float())

    def flow_matching_fwd2(self, batch, u_frac=None, noise_std=0.0):
        x_a, t_a, x_b, t_b, _, _ = batch            # x_*: [B,F,C,H,W], t_*: [B,F] (constant per sample)

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

    def flow_matching_fwd(self, batch, u_frac=0.0, noise_std=0.0):
        x_a, t_a, x_b, t_b, _, _ = batch
        assert t_a.dtype == torch.float32

        # sample interpolated point
        if u_frac is None:
            u_frac = torch.rand_like(t_a)
        else:
            u_frac = torch.full_like(t_a, u_frac)

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

        """
        # 3) model forward + loss
        with self.autocast_ctx:
            v_pred = self.core_fwd(x_u, s_u)
        return F.mse_loss(v_pred.float(), v_hat.float())  # TODO: scale by dt
        """
        with self.autocast_ctx:
            v_pred = self.core_fwd(x_u, s_u)
        # per-example MSE, then weight by |dt| (approximate ∫ over time)
        mse = F.mse_loss(v_pred.float(), v_hat.float(), reduction='none')
        per_ex = mse.reshape(mse.size(0), -1).mean(dim=1)      # [B]
        w = (dt ** 2) / (dt ** 2).mean()               # normalize weights
        return (w * per_ex).mean()

    def ayf_emd(
        self,
        batch,
        use_phi: bool = False,
        tangent_norm: bool = True,
        local_span: float = 0.05,
    ):
        """
        offline AYF-EMD (one-step Euler):
          s=(x_a,t_a)  u=(x_b,t_b) with t_a > t_b
          Compare: direct s→t  vs   two-step s→u→t (via-u under stop-grad)
          t is chosen ≤ u and ≤ s (forward in time for both paths)
        """
        x_a, t_a, x_b, t_b, _, _ = batch

        # ----- choose target time in RAW clock (not φ) -----
        # cap span by how much room remains to t=0
        # sample a strictly positive local step below u (since t_a > t_b)
        rho = torch.rand_like(t_b).clamp_min(1e-2)  # avoid ~0 step
        span_cap = torch.minimum(torch.full_like(t_b, local_span), t_b)
        t_raw = (t_b - rho * span_cap).clamp_min(0.0)  # strictly < t_b

        # ----- optional time reparameterization for conditioning only -----
        if use_phi:
            eps = 1e-4
            def phi(z: torch.Tensor) -> torch.Tensor:
                z = z.clamp(eps, 1 - eps)
                return torch.log(z / (1 - z))  # logit
            t_a_phi, t_b_phi, t_phi = phi(t_a), phi(t_b), phi(t_raw)
        else:
            t_a_phi, t_b_phi, t_phi = t_a, t_b, t_raw

        def expand_like(ts: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return ts.view(ts.shape + (1,) * (ref.ndim - ts.ndim))

        # RAW Δt for Euler steps; broadcast to match x_*
        dt_s = expand_like(t_raw - t_a, x_a).to(x_a.dtype)  # Δt from s to t (≤ 0)
        dt_u = expand_like(t_raw - t_b, x_b).to(x_b.dtype)  # Δt from u to t (≤ 0)

        # Optional per-sample tangent normalization
        def normalize_tangent(v: torch.Tensor) -> torch.Tensor:
            denom = v.pow(2).mean(dim=tuple(range(1, v.ndim)), keepdim=True).sqrt().clamp_min(1e-6)
            return v / denom.detach()  # supervise direction; stop scale-grad

        # ----- direct branch: s → t (grads on) -----
        with self.autocast_ctx:
            v_a = self.core_fwd(x_a, t_a_phi)
            if tangent_norm:
                v_a = normalize_tangent(v_a)
        # do the Euler math in fp32 for stability, then cast back
        x_direct = (x_a.float() + dt_s.float() * v_a.float()).to(x_a.dtype)

        # ----- via-u branch: u → t (STOP-GRAD) -----
        with torch.no_grad():
            with self.autocast_ctx:
                v_b = self.core_fwd(x_b, t_b_phi)
                if tangent_norm:
                    v_b = normalize_tangent(v_b)
            x_via_u = (x_b.float() + dt_u.float() * v_b.float()).to(x_b.dtype)

        return F.mse_loss(x_direct.float(), x_via_u.float())

    @torch.compile
    def core_fwd(self, *args, **kwargs):
        core = self.get_module(ema=False).core
        return core(*args, **kwargs)
