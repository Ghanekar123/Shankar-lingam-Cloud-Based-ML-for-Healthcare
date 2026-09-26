"""Multi-task loss combining classification + heteroscedastic regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskLoss(nn.Module):
    def __init__(self, w_diag=1.0, w_sev=1.0, w_det=1.0, w_risk=1.0):
        super().__init__()
        self.w = dict(diag=w_diag, sev=w_sev, det=w_det, risk=w_risk)

    def forward(self, out, batch):
        l_diag = F.cross_entropy(out["diag_logits"], batch["y_diag"])
        l_sev  = F.cross_entropy(out["sev_logits"],  batch["y_sev"])
        l_det  = F.binary_cross_entropy_with_logits(out["det_logit"], batch["y_det"])

        mu, logv = out["risk_mu"], out["risk_logv"]
        y = batch["y_risk"]
        l_risk = 0.5 * (torch.exp(-logv) * (y - mu) ** 2 + logv).mean()

        total = (self.w["diag"] * l_diag + self.w["sev"] * l_sev +
                 self.w["det"]  * l_det  + self.w["risk"] * l_risk)
        return total, {"diag": l_diag.item(), "sev": l_sev.item(),
                       "det": l_det.item(), "risk": l_risk.item()}