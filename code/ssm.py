"""Mamba-style State-Space temporal module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSSM(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2)
        self.x_proj   = nn.Linear(self.d_inner, self.d_state * 2 + 1)
        self.dt_proj  = nn.Linear(self.d_state * 2 + 1, self.d_inner)
        self.A_log    = nn.Parameter(torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
                 .unsqueeze(0).repeat(self.d_inner, 1)))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = F.silu(x_in)

        proj = self.x_proj(x_in)
        dt, Bp, Cp = proj.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(proj))
        A = -torch.exp(self.A_log)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            B_t  = Bp[:, t, :].unsqueeze(1)
            C_t  = Cp[:, t, :].unsqueeze(1)
            x_t  = x_in[:, t, :].unsqueeze(-1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))
            B_bar = dt_t * B_t
            h = A_bar * h + B_bar * x_t
            ys.append((h * C_t).sum(-1))
        y = torch.stack(ys, dim=1)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_in
        y = y * F.silu(z)
        return self.norm(self.out_proj(y)) + residual