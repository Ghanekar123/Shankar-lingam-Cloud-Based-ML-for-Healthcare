"""SAM-CRN: Secure Adaptive Multimodal Clinical Reasoning Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import ModalityEncoder
from graph_attention import GraphAttentionLayer
from ssm import SimpleSSM
from fusion import AdaptiveFusion
from knowledge_attention import KnowledgeGuidedAttention
from heads import UncertaintyHeads


class SAM_CRN(nn.Module):
    def __init__(self, in_dim, ehr_idx, iot_idx, physio_idx, lab_idx,
                 n_diag, n_sev, hidden=64):
        super().__init__()
        self.ehr_idx, self.iot_idx = ehr_idx, iot_idx
        self.physio_idx, self.lab_idx = physio_idx, lab_idx

        self.enc_ehr    = ModalityEncoder(len(ehr_idx),    hidden, hidden)
        self.enc_iot    = ModalityEncoder(len(iot_idx),    hidden, hidden)
        self.enc_physio = ModalityEncoder(len(physio_idx), hidden, hidden)
        self.enc_lab    = ModalityEncoder(len(lab_idx),    hidden, hidden)

        self.graph_attn = GraphAttentionLayer(hidden, hidden)
        self.ssm        = SimpleSSM(hidden, d_state=16, expand=2)
        self.fusion     = AdaptiveFusion(hidden)
        self.kga        = KnowledgeGuidedAttention(hidden, num_concepts=32)
        self.heads      = UncertaintyHeads(hidden, n_diag, n_sev)

    def forward(self, ehr, iot, physio, lab):
        h_ehr    = self.enc_ehr(ehr)
        h_iot    = self.enc_iot(iot)
        h_physio = self.enc_physio(physio)
        h_lab    = self.enc_lab(lab)

        nodes = torch.stack([h_ehr, h_iot, h_physio, h_lab], dim=1)

        with torch.no_grad():
            n = F.normalize(nodes, dim=-1)
            adj = torch.matmul(n, n.transpose(-2, -1))
            adj = (adj > 0.1).float()
            adj = adj + torch.eye(4, device=adj.device).unsqueeze(0)

        h_graph_seq, attn = self.graph_attn(nodes, adj)
        h_graph = h_graph_seq.mean(dim=1)

        h_ssm_seq = self.ssm(h_graph_seq)
        h_mamba   = h_ssm_seq.mean(dim=1)

        h_fused, alpha = self.fusion(h_graph, h_mamba)
        h_clin, kg_attn = self.kga(h_fused)

        diag_logits, sev_logits, det_logit, risk_mu, risk_logv = self.heads(h_clin)

        return {
            "diag_logits": diag_logits,
            "sev_logits":  sev_logits,
            "det_logit":   det_logit,
            "risk_mu":     risk_mu,
            "risk_logv":   risk_logv,
            "alpha":       alpha,
            "graph_attn":  attn,
            "kg_attn":     kg_attn,
            "embedding":   h_clin,
        }