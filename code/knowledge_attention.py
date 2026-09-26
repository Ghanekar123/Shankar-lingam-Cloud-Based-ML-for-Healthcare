"""Knowledge-Guided Clinical Attention over learned concepts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeGuidedAttention(nn.Module):
    def __init__(self, dim, num_concepts=32):
        super().__init__()
        self.concept_emb = nn.Parameter(torch.randn(num_concepts, dim) * 0.02)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        q = self.q(x).unsqueeze(1)
        k = self.k(self.concept_emb).unsqueeze(0)
        v = self.v(self.concept_emb).unsqueeze(0)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        ctx = (attn @ v).squeeze(1)
        return self.norm(x + ctx), attn.squeeze(1)