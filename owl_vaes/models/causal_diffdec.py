"""
Diffusion decoder for frame pairs

Idea:
By diffusing two frames together, we encourage model
to generate in a way that makes use of other frames latent,
such that the generation is consistent.
"""

import torch
from torch import nn
import torch.nn.functional as F
import math

from copy import deepcopy

from ..utils import freeze
from ..configs import TransformerConfig
from ..nn.resnet import SquareToLandscape, LandscapeToSquare

from ..nn.dit import DiT, FinalLayer
from ..nn.embeddings import LearnedPosEnc
from ..nn.embeddings import TimestepEmbedding, StepEmbedding

from .diffdec import DiffusionDecoderCore

class PairDiffDecCore(DiffusionDecoderCore):
    def __init__(self, config : TransformerConfig):
        super().__init__(config)

        self.pos_enc_prev_z = LearnedPosEnc(config.latent_size**2, config.d_model)

    def flatten(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, self.n_p_y, self.p, self.n_p_x, self.p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b, self.n_p_y * self.n_p_x, self.p * self.p * c)
    
    def flatten_latent(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

    def forward(self, x, z_x, z_y, ts, y = None):
        # x is [b,c,h,w] (frame)
        # y is [b,c,h,w] (prev frame)
        # z_x is [b,c,h,w] (latent of x)
        # z_y is [b,c,h,w] (latent of y)

        cond = self.ts_embed(ts)

        x = self.flatten(x)
        x = self.proj_in(x)
        x = self.pos_enc_x(x)

        z_x = self.flatten_latent(z_x)
        z_y = self.flatten_latent(z_y)
        z_x = self.proj_in_z(z_x)
        z_y = self.proj_in_z(z_y)
        z_x = self.pos_enc_z(z_x)
        z_y = self.pos_enc_z(z_y)

        n = x.shape[1]
        x = torch.cat([x,z_x,self.pos_enc_prev_z(z_y)],dim=1)
        x = self.blocks(x, cond)
        x = x[:,:n]
        x = self.final(x, cond)

        if y is not None:
            y = self.flatten(y)
            y = self.proj_in(y)
            y = self.pos_enc_x(y)
            y = torch.cat([y,z_y,self.pos_enc_prev_z(z_x)],dim=1)
            y = self.blocks(y, cond)
            y = y[:,:n]
            y = self.final(y, cond)
            return x,y
        else:
            return x

class PairDiffDec(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = PairDiffDecCore(config)
    
    def forward(self, x, y, z_x, z_y):
        """
        For training we assume y is always given
        """
        with torch.no_grad():
            ts = torch.randn(len(x), device = x.device, dtype = x.dtype).sigmoid()

            eps_x = torch.randn_like(x)
            eps_y = torch.randn_like(y)
            ts_exp = ts.view(-1, 1, 1, 1).expand_as(x)

            lerpd_x = x * (1. - ts_exp) + ts_exp * eps_x
            lerpd_y = y * (1. - ts_exp) + ts_exp * eps_y

            target_x = eps_x - x
            target_y = eps_y - y

        pred_x, pred_y = self.core(lerpd_x, z_x, z_y, ts, y)

        diff_loss_x = F.mse_loss(pred_x, target_x)
        diff_loss_y = F.mse_loss(pred_y, target_y)

        return 0.5 * (diff_loss_x + diff_loss_y)
    