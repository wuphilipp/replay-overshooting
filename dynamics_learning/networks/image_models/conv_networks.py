from typing import List, Sequence, Tuple

import numpy as np
import torch
from fannypack.nn import resblocks
from torch import nn as nn

# Modified from: https://github.com/utiasSTARS/robust-latent-srl/blob/master/srl/srl/train.py

# ----------------- #
# AUXILIARY CLASSES #
# ----------------- #


# TODO document these classes


class _Reshape(nn.Module):
    def __init__(self, shape: Sequence[int]) -> None:
        super(_Reshape, self).__init__()
        self._shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*self._shape)


class _Conv2dLayerNorm(nn.Module):
    def __init__(
        self, c_in, c_out, kernel_size, h_out, w_out, batch_norm=True, **kwargs
    ):
        super(_Conv2dLayerNorm, self).__init__()
        self._conv = nn.Conv2d(c_in, c_out, kernel_size, **kwargs)
        self._act = nn.LeakyReLU()
        self._resblock = resblocks.Conv2d(c_out, activation="leaky_relu")
        if batch_norm:
            self._ln = nn.BatchNorm2d(c_out)
            self._ln_post = nn.BatchNorm2d(c_out)
        else:
            self._ln = nn.LayerNorm([c_out, h_out, w_out])
            self._ln_post = nn.LayerNorm([c_out, h_out, w_out])

    def forward(self, x):
        res = self._act(self._ln(self._conv(x)))
        return self._resblock(self._ln_post(res))


class _ConvT2dLayerNorm(nn.Module):
    def __init__(
        self, c_in, c_out, kernel_size, h_out, w_out, batch_norm=True, **kwargs
    ):
        super(_ConvT2dLayerNorm, self).__init__()
        self._convt = nn.ConvTranspose2d(c_in, c_out, kernel_size, **kwargs)
        self._act = nn.LeakyReLU()
        self._resblock = resblocks.Conv2d(c_out, activation="leaky_relu")
        if batch_norm:
            self._ln = nn.BatchNorm2d(c_out)
            self._ln_post = nn.BatchNorm2d(c_out)
        else:
            self._ln = nn.LayerNorm([c_out, h_out, w_out])
            self._ln_post = nn.LayerNorm([c_out, h_out, w_out])

    def forward(self, x):
        res = self._act(self._ln(self._convt(x)))
        return self._resblock(self._ln_post(res))


def get_simple_encoder_and_dsd_decoder(
    in_channels: int,
    network_channels: List[int],
    img_size: int,
    latent_dim: int,
    pixel_res: int,
    cond_channels: int = 0,
) -> Tuple[nn.Module, nn.Module]:
    """Get and encoder and decoder for use in a VAE."""
    # Construct Encoder
    # Output is diagonal covariance Gaussian
    layers: List[nn.Module] = []
    in_c = in_channels + cond_channels
    for i, c in enumerate(network_channels):
        layers.append(
            _Conv2dLayerNorm(
                in_c,
                c,
                3,
                img_size // (2 ** (i + 1)),
                img_size // (2 ** (i + 1)),
                padding=1,
                stride=2,
            )
        )
        in_c = c
    layers.append(nn.Flatten(start_dim=-3))  # out: (..., in_c)
    layers.append(nn.Linear(in_c, 2 * latent_dim))
    encoder = nn.Sequential(*layers)

    # Construct Decoder
    # Output is discrete log-softmax distribution
    # > see: "Pixel Recurrent Neural Networks", sec. 5.3
    div_base = 2
    factor_list = reversed(range(1, int(np.log(pixel_res) / np.log(div_base))))
    pix_layers = [
        _ConvT2dLayerNorm(
            pixel_res // div_base ** (i + 1),
            pixel_res // div_base ** i,
            3,
            img_size,
            img_size,
            padding=1,
        )
        for i in factor_list
    ]
    decoder = nn.Sequential(
        nn.Linear(latent_dim + cond_channels, 4 * img_size * img_size),
        _Reshape((-1, 64, int(img_size / 4), int(img_size / 4))),
        nn.LeakyReLU(),
        nn.LayerNorm([64, img_size // 4, img_size // 4]),
        _ConvT2dLayerNorm(64, 16, 2, img_size // 2, img_size // 2, stride=2),
        _ConvT2dLayerNorm(16, 1, 2, img_size, img_size, stride=2),
        *pix_layers,
        nn.ConvTranspose2d(pixel_res // div_base, pixel_res, 3, padding=1),
        nn.LeakyReLU(),
        nn.LayerNorm([pixel_res, img_size, img_size]),
        resblocks.Conv2d(pixel_res, pixel_res * div_base, activation="leaky_relu"),
        nn.LogSoftmax(dim=-3),
    )
    return encoder, decoder
