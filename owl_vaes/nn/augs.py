"""
Augmentations for paired image data
"""

import torch
from torch import nn
import torch.nn.functional as F
from kornia.geometry.transform import Affine


class PairedRandomAffine(nn.Module):
    def __init__(
        self,
        scale_range=0.05,  # Scale range (e.g. 0.05 means 0.95-1.05x)
        shift_range=0.1,   # Shift range as fraction of image size
        shear_range=0.25   # Shear range in degrees
    ):
        super().__init__()
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.shear_range = shear_range

    def _rand_uniform(self, batch_size, device, dtype):
        """Helper to generate random uniform values in [-1,1]"""
        return 2.0 * torch.rand(batch_size, device=device, dtype=dtype) - 1.0

    def _get_affine_transform(self, x):
        """Returns an Affine transform with random parameters"""
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Sample random parameters
        rand = lambda: self._rand_uniform(batch_size, device, dtype)

        scale_x = rand() * self.scale_range + 1.0
        scale_y = rand() * self.scale_range + 1.0
        shift_x = rand() * self.shift_range
        shift_y = rand() * self.shift_range
        shear_x = rand() * self.shear_range
        shear_y = rand() * self.shear_range

        # Random horizontal flip
        flip = (torch.rand(batch_size, device=device) < 0.5).float() * 2 - 1
        scale_x = scale_x * flip

        return Affine(
            scale_factor=torch.stack([scale_x, scale_y], dim=-1),
            translation=torch.stack([shift_x, shift_y], dim=-1),
            shear=torch.stack([shear_x, shear_y], dim=-1),
            padding_mode='reflection'
        )

    def _transform_image(self, x, affine_fn):
        """Applies transform with up/downsampling"""
        h, w = x.shape[-2:]
        x = x.float()

        # Upsample, transform, downsample
        x = F.interpolate(x, size=(h*2, w*2), mode='bicubic', align_corners=True)
        x = affine_fn(x)
        x = F.interpolate(x, size=(h, w), mode='bicubic', align_corners=True)

        return x

    def forward(self, x_real, x_fake=None):
        orig_dtype = x_real.dtype

        # Get transform and apply to real image
        affine_fn = self._get_affine_transform(x_real.float())
        x_real_aug = self._transform_image(x_real, affine_fn)

        if x_fake is None:
            return x_real_aug.to(orig_dtype)

        # Apply same transform to fake image
        x_fake_aug = self._transform_image(x_fake, affine_fn)
        return x_real_aug.to(orig_dtype), x_fake_aug.to(orig_dtype)
