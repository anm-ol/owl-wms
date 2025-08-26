import torch
import torch.nn.functional as F

from .world_trainer import WorldTrainer


class RFTPairDistillTrainer(WorldTrainer):
    def fwd_step(self, batch):
        return self.diffusion_forcing_distillation(batch)

    def diffusion_forcing_distillation(self, batch):
        xs, t = batch["x_samples"], batch["times"]                 # xs: [B,N,K,C,H,W], t: [K] or [B,N,K] (ascending)
        B, N, K, C, H, W = xs.shape
        assert K >= 2, "Need at least two samples per path to bracket ts."

        # Normalize times to [B,N,K]
        t = t.view(B, 1, K).expand(B, N, K)

        # Time arithmetic in fp32; keep latents in their native dtype (fp16/bf16)
        t32 = t.to(device=xs.device, dtype=torch.float32)

        with torch.no_grad():
            # Match other methods' sampling: ts ~ sigmoid(N(0,1))) with shape [B,N]
            ts = torch.randn(B, N, device=xs.device, dtype=xs.dtype).sigmoid()
            ts32 = ts.to(torch.float32)

            # Batched neighbor lookup along K (dim=2): t_lo <= ts < t_hi
            hi = torch.searchsorted(t32, ts32.unsqueeze(-1), right=True).clamp_(1, K - 1)  # [B,N,1]
            lo = hi - 1

            # Gather neighbor times along dim=2
            t_lo = torch.take_along_dim(t32, lo, dim=2).squeeze(2)  # [B,N]
            t_hi = torch.take_along_dim(t32, hi, dim=2).squeeze(2)  # [B,N]

            denom = (t_hi - t_lo).abs().clamp_min(torch.finfo(torch.float32).eps)
            w = ((ts32 - t_lo) / denom).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,N,1,1,1]

            # Gather neighbor samples along K with a single-dim gather (fast & coalesced)
            idx_lo = lo.expand(B, N, 1, C, H, W)
            idx_hi = hi.expand(B, N, 1, C, H, W)
            x_lo = torch.take_along_dim(xs, idx_lo, dim=2).squeeze(2)  # [B,N,C,H,W]
            x_hi = torch.take_along_dim(xs, idx_hi, dim=2).squeeze(2)  # [B,N,C,H,W]

            # Interpolate to x_t at ts
            x_t = torch.lerp(x_lo, x_hi, w.to(dtype=xs.dtype))         # keep dtype same as xs

            # Endpoint velocity target (unchanged)
            v_target = xs[:, :, -1] - xs[:, :, 0]                      # [B,N,C,H,W]

        with self.autocast_ctx:
            v_pred = self.core_fwd(x_t.contiguous(), ts)               # .contiguous() can help memory access
        return F.mse_loss(v_pred.float(), v_target.float())

    def standard_loss_teacher(self, batch):
        x0 = batch["x_clean"]                                     # [B, N, C, H, W]
        B, N = x0.size(0), x0.size(1)

        assert torch.all(batch["time_clean"] == 0.0)
        assert torch.all(batch["time_noise"] == 1.0)

        with torch.no_grad():
            ts = torch.randn(B, N, device=x0.device, dtype=x0.dtype).sigmoid()
            x1 = batch["x_noise"]  # teachers input noise
            x_t = x0 + (x1 - x0) * ts.view(B, N, 1, 1, 1)  # lerp to noise level @ ts
            v_target = x1 - x0

        with self.autocast_ctx:
            v_pred = self.core_fwd(x_t, ts)
        return F.mse_loss(v_pred.float(), v_target.float())

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


    def to_clean_via_flow(self, batch):
        x_t, t = batch["x_a"], batch["time_a"]
        x0      = batch["x_clean"]
        with self.autocast_ctx:
            v = self.core_fwd(x_t, t.float())             # predicts displacement
            x0_pred = x_t - t[...,None,None,None] * v
        return F.mse_loss(x0_pred.float(), x0.float())

    def consistency_step(self, batch, dt_eps=1e-6):
        x_a, t_a = batch["x_a"], batch["time_a"]          # [B,F,C,H,W], [B,F]
        x_b, t_b = batch["x_b"], batch["time_b"]

        dt = (t_a - t_b).clamp_min(dt_eps)[..., None, None, None]
        with self.autocast_ctx:
            v = self.core_fwd(x_a, t_a.float())           # your head predicts *displacement* (RF)
            x_pred = x_a - dt * v                         # Euler update using student's v
        return F.mse_loss(x_pred.float(), x_b.float())

    def rf_clean_to_step(self, batch, u_eps: float = 0.02):
        """
        Rectified-Flow (displacement) loss on the truncated horizon [0, t_a]:
          endpoints: (x_clean, 0) -> (x_a, t_a)
          input:     x_u at exact time s in (0, t_a)
          target:    x_a - x_clean   (displacement; no /dt)
        """
        x0, t0 = batch["x_clean"], batch["time_clean"]      # t0 == 0
        xa, ta = batch["x_a"],     batch["time_a"]          # ta in (0, 1]

        assert torch.all(ta != t0)

        # sample absolute time s ∈ (ε*ta, (1-ε)*ta)  (timestep-exact, no extra noise)
        s01 = torch.rand_like(ta).clamp(u_eps, 1.0 - u_eps)
        s   = (s01 * ta).float()                            # keep times in fp32

        # place state exactly at time s on the straight chord clean→x_a
        lam = (s / ta).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,F,1,1,1]
        xu  = x0 + (xa - x0) * lam

        v_hat = xa - x0                                     # displacement target (like GameRFT form)
        with self.autocast_ctx:
            v_pred = self.core_fwd(xu, s)                   # t in fp32 into the embedding
        return F.mse_loss(v_pred.float(), v_hat.float())

    def fixed_rectified_flow_teacher(self, batch, u_frac: float | None = None, noise_std: float = 0.0, u_eps: float = 0.02):
        x_a, t_a = batch["x_a"], batch["time_a"]      # earlier (noisier) state, higher t
        x_b, t_b = batch["x_b"], batch["time_b"]      # later (cleaner) state, lower t

        # λ ∈ (u_eps, 1-u_eps) to avoid endpoint conflicts at (x_a, t_a) or (x_b, t_b)
        u = torch.rand_like(t_a) if u_frac is None else torch.full_like(t_a, float(u_frac))
        if u_eps > 0: u = u.clamp(u_eps, 1.0 - u_eps)

        lam = u[..., None, None, None]
        x_u = x_a + (x_b - x_a) * lam
        if noise_std:
            x_u = x_u + noise_std * torch.randn_like(x_u)

        s_u = (t_a + (t_b - t_a) * u).float()        # keep times fp32

        # >>> key change: target should point NOISEWARD to match the sampler’s minus sign
        v_hat = x_a - x_b                             # not (x_b - x_a)

        with self.autocast_ctx:
            v_pred = self.core_fwd(x_u, s_u)
        return F.mse_loss(v_pred.float(), v_hat.float())

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
