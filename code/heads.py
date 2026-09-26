"""Multi-task heads with uncertainty (aleatoric) for regression."""

import torch
import torch.nn as nn


class UncertaintyHeads(nn.Module):
    def __init__(self, dim, n_diag, n_sev):
        super().__init__()
        self.head_diag = nn.Linear(dim, n_diag)
        self.head_sev  = nn.Linear(dim, n_sev)
        self.head_det  = nn.Linear(dim, 1)
        self.head_risk_mu   = nn.Linear(dim, 1)
        self.head_risk_logv = nn.Linear(dim, 1)

    def forward(self, x):
        diag_logits = self.head_diag(x)
        sev_logits  = self.head_sev(x)
        det_logit   = self.head_det(x).squeeze(-1)
        mu   = torch.sigmoid(self.head_risk_mu(x)).squeeze(-1)
        logv = torch.clamp(self.head_risk_logv(x).squeeze(-1), -6.0, 4.0)
        return diag_logits, sev_logits, det_logit, mu, logv