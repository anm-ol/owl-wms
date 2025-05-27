import os
import torch

class DeviceManager:
    envvar = os.getenv("DEVICE_TYPE")
    _device: str = envvar if isinstance(envvar, str) else "cuda"

    @classmethod
    def set_device(cls, device: str):
        torch.set_default_device(device)
        cls._device = device

    @classmethod
    def get_device(cls) -> str:
        if cls._device is None:
            raise ValueError("Device not initialized")
        return cls._device
