"""YOLOv1 architecture.

Two variants, both ending in the paper's FC head (hidden_dim -> 1470 ->
7x7x30, dropout 0.5 after the first FC):

* ``full``  : the paper's 24-conv GoogLeNet-inspired trunk with 1x1 reduction
              layers, or optionally a torchvision GoogLeNet conv trunk when
              ``pretrained=True``.
* ``fast``  : a 9-conv trunk (Fast YOLO) for quick CPU / Colab iteration.

Note: the paper text says "24 convolutional layers" while its Figure 3 diagram
is usually read as 21; this implementation uses the 24-conv reading. BatchNorm
is a configurable add-on (helps stability a lot); set ``batchnorm: false`` to
reproduce the paper's dropout-only conv stack exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_LEAK = 0.1


def _conv(
    in_ch: int, out_ch: int, kernel: int, stride: int, pad: int, batchnorm: bool = True
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=not batchnorm)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(_LEAK, inplace=True))
    return nn.Sequential(*layers)


class YOLOv1(nn.Module):
    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        variant: str = "full",
        pretrained: bool = False,
        dropout: float = 0.5,
        hidden_dim: int = 4096,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.variant = variant
        self.pretrained = pretrained

        self.features = self._build_features(variant, pretrained, batchnorm)
        flat_dim = 1024 * S * S
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.LeakyReLU(_LEAK, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, S * S * (B * 5 + C)),
        )

    # ------------------------------------------------------------------ #
    def _build_features(self, variant: str, pretrained: bool, batchnorm: bool) -> nn.Module:
        if pretrained:
            return self._pretrained_trunk()
        if variant == "fast":
            return self._fast_trunk(batchnorm)
        return self._full_trunk(batchnorm)

    @staticmethod
    def _full_trunk(bn: bool) -> nn.Sequential:
        c = lambda i, o, k, s, p: _conv(i, o, k, s, p, bn)  # noqa: E731
        return nn.Sequential(
            c(3, 64, 7, 2, 3),               # 448 -> 224
            nn.MaxPool2d(2, 2),              # -> 112
            c(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),              # -> 56
            c(192, 128, 1, 1, 0),
            c(128, 256, 3, 1, 1),
            c(256, 256, 1, 1, 0),
            c(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),              # -> 28
            c(512, 256, 1, 1, 0),
            c(256, 512, 3, 1, 1),
            c(512, 256, 1, 1, 0),
            c(256, 512, 3, 1, 1),
            c(512, 256, 1, 1, 0),
            c(256, 512, 3, 1, 1),
            c(512, 256, 1, 1, 0),
            c(256, 512, 3, 1, 1),
            c(512, 256, 1, 1, 0),
            c(256, 512, 3, 1, 1),
            c(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),              # -> 14
            c(1024, 512, 1, 1, 0),
            c(512, 1024, 3, 1, 1),
            c(1024, 512, 1, 1, 0),
            c(512, 1024, 3, 1, 1),
            c(1024, 1024, 3, 2, 1),          # -> 7
            c(1024, 1024, 3, 1, 1),
            c(1024, 1024, 3, 1, 1),
        )

    @staticmethod
    def _fast_trunk(bn: bool) -> nn.Sequential:
        c = lambda i, o, k, s, p: _conv(i, o, k, s, p, bn)  # noqa: E731
        return nn.Sequential(
            c(3, 64, 7, 2, 3),               # 448 -> 224
            nn.MaxPool2d(2, 2),              # -> 112
            c(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),              # -> 56
            c(192, 128, 1, 1, 0),
            c(128, 256, 3, 1, 1),
            c(256, 256, 1, 1, 0),
            c(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),              # -> 28
            c(512, 256, 1, 1, 0),
            c(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),              # -> 14
            c(512, 1024, 3, 2, 1),           # -> 7
        )

    @staticmethod
    def _pretrained_trunk() -> nn.Sequential:
        from torchvision.models import GoogLeNet_Weights, googlenet

        try:
            net = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        except Exception:  # noqa: BLE001 - offline fallback
            net = googlenet(weights=None)
            print("[model] WARNING: could not fetch GoogLeNet weights, using random init")
        return nn.Sequential(net.features, nn.AdaptiveMaxPool2d(7))

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)


def build_model(cfg, pretrained: bool | None = None) -> YOLOv1:
    m = cfg.model
    if pretrained is None:
        pretrained = m.pretrained
    return YOLOv1(
        S=m.S,
        B=m.B,
        C=m.C,
        variant=m.variant,
        pretrained=pretrained,
        dropout=m.dropout,
        hidden_dim=m.hidden_dim,
        batchnorm=m.batchnorm,
    )
