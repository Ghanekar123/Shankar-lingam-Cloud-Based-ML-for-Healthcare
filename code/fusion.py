"""Adaptive Graph-Temporal Fusion with learnable gate alpha."""

import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Linear(dim, 1), nn.Sigmoid()
        )

    def forward(self, h_graph, h_mamba):
        alpha = self.gate(torch.cat([h_graph, h_mamba], -1))
        return alpha * h_graph + (1 - alpha) * h_mamba, alpha