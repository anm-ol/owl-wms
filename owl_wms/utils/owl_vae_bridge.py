import sys
import os

import torch
from diffusers import AutoencoderDC, AutoencoderKLLTXVideo, AutoencoderKLWan


sys.path.append("./owl-vaes")
from owl_vaes.utils.proxy_init import load_proxy_model
from owl_vaes.models import get_model_cls
from owl_vaes.configs import Config

def _get_decoder_only():
    model = load_proxy_model(
        "../checkpoints/128x_proxy_titok.yml",
        "../checkpoints/128x_proxy_titok.pt",
        "../checkpoints/16x_dcae.yml",
        "../checkpoints/16x_dcae.pt"
    )
    del model.transformer.encoder
    return model

def get_decoder_only(vae_id, cfg_path, ckpt_path):
        if vae_id == "dcae":
            model_id = "mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers"
            model = AutoencoderDC.from_pretrained(model_id).bfloat16().cuda().eval()
            del model.encoder
            return model.decoder
        if vae_id == "ltx":
            model = AutoencoderKLLTXVideo.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).cuda().eval()
            del model.encoder
            return model.decoder
        if vae_id == "wan":
            model = AutoencoderKLWan.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).cuda().eval()
            del model.encoder
            return model.decoder
        else:
            cfg = Config.from_yaml(cfg_path).model
            model = get_model_cls(cfg.model_id)(cfg)
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
            except:
                model.decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
            del model.encoder
            model = model.decoder
            model = model.bfloat16().cuda().eval()
            return model

@torch.no_grad()
def make_batched_decode_fn(decoder, batch_size = 8, temporal_vae=True):
    def decode(x):
        # x is [b,n,c,h,w]
        b,n,c,h,w = x.shape
        if not temporal_vae:
            x = x.view(b*n,c,h,w).contiguous()
        else:
            x = x.permute(0, 2, 1, 3, 4)
            x = x.contiguous()
        batches = x.split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b*n,c,h,w]
        if not temporal_vae:
            _,c,h,w = x.shape
            x = x.view(b,n,c,h,w).contiguous()
        else:
            x = x.contiguous()

        return x
    return decode

@torch.no_grad()
def make_batched_audio_decode_fn(decoder, batch_size = 8):
    def decode(x):
        # x is [b,n,c] audio samples
        x = x.transpose(1,2)
        b,c,n = x.shape

        batches = x.contiguous().split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b,c,n]
        x = x.transpose(-1,-2).contiguous() # [b,n,2]

        return x
    return decode