"""A small CNN for data labeled with ImSpec Labeler.

One class covers both kinds of data: 2D convolutions for images and photos, 1D for spectra.
The scan-size / bias-range numbers (aux) go through a small side branch and are joined to the
image features before the last layers.

    input (n, C, H, W) or (n, C, P)
      -> [Conv -> BatchNorm -> ReLU -> MaxPool] x len(channels)
      -> global average pooling                      (so any input size works)
      -> join with the aux branch (Linear -> ReLU)   (if the dataset has aux values)
      -> Linear -> ReLU -> Dropout -> Linear         -> class scores, or one number

Everything that defines the network is in one dict (``config["model"]``), saved in model_config.txt,
so the same network can be built again for retraining or inference.
"""
import math

import torch
from torch import nn


def default_model_config(kind, sample_shape, n_aux, n_outputs):
    """Network settings for one kind of data. ``sample_shape``: (C, H, W) or (C, P)."""
    dims = 1 if kind == "spectrum" else 2
    smallest = min(sample_shape[1:])
    # every block halves the size: do not pool a small input down to nothing
    n_blocks = max(1, min(3, int(math.log2(max(smallest, 2)))))
    return {
        "dims": dims,                              # 1: Conv1d (spectra), 2: Conv2d (images)
        "in_channels": int(sample_shape[0]),
        "n_aux": int(n_aux),
        "n_outputs": int(n_outputs),               # number of classes, or 1 for a score
        "channels": [16, 32, 64][:n_blocks],
        "kernel_size": 5 if dims == 1 else 3,
        "hidden": 64,
        "aux_hidden": 16,
        "dropout": 0.3,
    }


class LabelCNN(nn.Module):
    def __init__(self, dims, in_channels, n_aux, n_outputs, channels=(16, 32, 64), kernel_size=3,
                 hidden=64, aux_hidden=16, dropout=0.3):
        super().__init__()
        if dims not in (1, 2):
            raise ValueError("dims must be 1 (spectra) or 2 (images)")
        Conv, Norm, Pool = (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d) if dims == 1 else \
            (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d)
        layers, previous = [], in_channels
        for width in channels:
            layers += [Conv(previous, width, kernel_size, padding=kernel_size // 2), Norm(width),
                       nn.ReLU(inplace=True), Pool(2)]
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1) if dims == 1 else nn.AdaptiveAvgPool2d(1)
        self.n_aux = n_aux
        self.aux = nn.Sequential(nn.Linear(n_aux, aux_hidden), nn.ReLU(inplace=True)) if n_aux else None
        self.head = nn.Sequential(
            nn.Linear(previous + (aux_hidden if n_aux else 0), hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_outputs))

    def forward(self, x, aux=None):
        h = self.pool(self.features(x)).flatten(1)
        if self.aux is not None:
            if aux is None:
                raise ValueError("this model was built with aux values: pass them as the second argument")
            h = torch.cat([h, self.aux(aux)], dim=1)
        return self.head(h)


def build_model(model_config):
    """The network described by ``config["model"]`` (see default_model_config)."""
    c = dict(model_config)
    return LabelCNN(c["dims"], c["in_channels"], c["n_aux"], c["n_outputs"], tuple(c["channels"]),
                    c["kernel_size"], c["hidden"], c["aux_hidden"], c["dropout"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
