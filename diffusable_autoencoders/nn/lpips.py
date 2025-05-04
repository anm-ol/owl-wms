import torch 
from torch import nn
import torch.nn.functional as F

from lpips import LPIPS # Make sure this is my custom LPIPS, not pip install LPIPS
from .augs import PairedRandomAffine

# vgg takes 224 sized images
def vgg_patchify(x):
    _,_,h,_ = x.shape
    if h != 512: x = F.interpolate(x, (512, 512), mode='bicubic', align_corners=True)
    
    patch_size = 224
    tl = x[:,:,:patch_size,:patch_size]
    tr = x[:,:,:patch_size,-patch_size:]
    bl = x[:,:,-patch_size:,:patch_size] 
    br = x[:,:,-patch_size:,-patch_size:]
    return torch.cat([tl, tr, bl, br], dim=0)

class VGGLPIPS(nn.Module):
    def __init__(self):
        super().__init__()

        self.aug = PairedRandomAffine()
        self.model = LPIPS(net='vgg')

    def forward(self, x_fake, x_real):
        x_fake, x_real = self.aug(x_fake, x_real)
        _,_,h,_ = x_fake.shape
        if h > 224:
            x_fake = vgg_patchify(x_fake)
            x_real = vgg_patchify(x_real)
        
        return self.model(x_fake, x_real).mean()
        
        