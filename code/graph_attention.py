"""Graph Attention Layer over clinical modality graph."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)
        B, N, D = Wh.shape
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, D)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, D)
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], -1)).squeeze(-1))
        mask = (adj <= 0).float() * -1e9
        e = e + mask
        attn = F.softmax(e, dim=-1)
        out = torch.matmul(attn, Wh)
        return out, attn