"""Isolation Forest anomaly detection on learned embeddings."""

import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from config import DEVICE, SEED


def get_embeddings(model, loader):
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            o = model(batch["ehr"], batch["iot"], batch["physio"], batch["lab"])
            embs.append(o["embedding"].cpu().numpy())
    return np.concatenate(embs)


def run_anomaly_detection(model, train_loader, test_loader, contamination=0.05):
    emb_tr = get_embeddings(model, train_loader)
    emb_te = get_embeddings(model, test_loader)
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED)
    iso.fit(emb_tr)
    pred_anom = iso.predict(emb_te)
    n_flag = (pred_anom == -1).sum()
    print(f"\nAnomaly detection: {n_flag} / {len(pred_anom)} test samples flagged as anomalous")
    return pred_anom, iso