"""PyTorch dataset for multimodal healthcare data."""

import torch
from torch.utils.data import Dataset


class MultimodalHealthcareDataset(Dataset):
    def __init__(self, df, feature_cols, ehr_cols, iot_cols, physio_cols, lab_cols):
        self.df = df.reset_index(drop=True)
        self.feature_cols = [c for c in feature_cols if c in df.columns]

        self.ehr_idx    = [self.feature_cols.index(c) for c in ehr_cols    if c in self.feature_cols]
        self.iot_idx    = [self.feature_cols.index(c) for c in iot_cols    if c in self.feature_cols]
        self.physio_idx = [self.feature_cols.index(c) for c in physio_cols if c in self.feature_cols]
        self.lab_idx    = [self.feature_cols.index(c) for c in lab_cols    if c in self.feature_cols]

        self.X      = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
        self.y_diag = torch.tensor(df["diagnosis_class"].values,     dtype=torch.long)
        self.y_risk = torch.tensor(df["risk_score"].values,          dtype=torch.float32)
        self.y_sev  = torch.tensor(df["severity_class"].values,      dtype=torch.long)
        self.y_det  = torch.tensor(df["deterioration_label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {
            "x":      self.X[i],
            "ehr":    self.X[i, self.ehr_idx]    if self.ehr_idx    else self.X[i],
            "iot":    self.X[i, self.iot_idx]    if self.iot_idx    else self.X[i],
            "physio": self.X[i, self.physio_idx] if self.physio_idx else self.X[i],
            "lab":    self.X[i, self.lab_idx]    if self.lab_idx    else self.X[i],
            "y_diag": self.y_diag[i],
            "y_risk": self.y_risk[i],
            "y_sev":  self.y_sev[i],
            "y_det":  self.y_det[i],
        }