import torch
from torch import nn

import time

def freeze(module : nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module : nn.Module):
    for param in module.parameters():
        param.requires_grad = True

class Timer:
    def reset(self):
        self.start_time = time.time()
    
    def hit(self):
        return time.time() - self.start_time
