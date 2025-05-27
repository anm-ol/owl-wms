from torch import nn
import torch
from torchtyping import TensorType  # type: ignore[import-untyped]

from owl_vaes.data.audio_loader import get_loader
from owl_vaes.nn.audio_blocks import (
    DecoderBlock,
    EncoderBlock,
    WNConv1d,
    get_activation,
)
from owl_vaes.utils.get_device import DeviceManager

device = DeviceManager.get_device()

class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] = [1, 2, 4, 8],
        strides: list[int] = [2, 4, 8, 8],
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]

        for i in range(self.depth - 1):
            layers += [
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[-1] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[-1] * channels,
                out_channels=latent_dim,
                kernel_size=3,
                padding=1,
            ),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
    ):
        super().__init__()
        self.out_channels = out_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            ),
        ]

        for i in range(self.depth - 1, 0, -1):
            layers += [
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[0] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[0] * channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.Tanh() if final_tanh else nn.Identity(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self.layers(x)


if __name__ == "__main__":
    loader = get_loader(1, "my_data/")
    sample = next(iter(loader))[0]

    myEncoder = OobleckEncoder(2, 128, use_snake=True, antialias_activation=True)
    myDecoder = OobleckDecoder(
        2, 128, antialias_activation=True, use_nearest_upsample=True
    )

    latent = myEncoder(sample)
    decoded = myDecoder(latent)

    print("Latent: ", latent)
    print("Latent: ", latent.shape)
    print("Latent: ", latent.dtype)

    print("Decoded: ", decoded)
    print("Decoded: ", decoded.shape)
    print("Decoded: ", decoded.dtype)

    print("MSE:", torch.mean((decoded - sample).pow(2)))
