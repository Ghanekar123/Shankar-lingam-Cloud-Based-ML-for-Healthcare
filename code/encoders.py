"""Per-modality MLP encoder."""

import torch.nn as nn


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)