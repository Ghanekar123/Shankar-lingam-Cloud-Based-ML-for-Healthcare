"""Evaluation utilities with class-name reporting."""

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

from config import DEVICE


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    buf = {"diag_true": [], "diag_pred": [], "diag_prob": [],
           "sev_true":  [], "sev_pred":  [], "sev_prob":  [],
           "det_true":  [], "det_prob":  [],
           "risk_true": [], "risk_pred": [], "risk_logv": []}
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(DEVICE)
        out = model(batch["ehr"], batch["iot"], batch["physio"], batch["lab"])
        buf["diag_prob"].append(F.softmax(out["diag_logits"], -1).cpu().numpy())
        buf["diag_pred"].append(out["diag_logits"].argmax(-1).cpu().numpy())
        buf["diag_true"].append(batch["y_diag"].cpu().numpy())
        buf["sev_prob"].append(F.softmax(out["sev_logits"], -1).cpu().numpy())
        buf["sev_pred"].append(out["sev_logits"].argmax(-1).cpu().numpy())
        buf["sev_true"].append(batch["y_sev"].cpu().numpy())
        buf["det_prob"].append(torch.sigmoid(out["det_logit"]).cpu().numpy())
        buf["det_true"].append(batch["y_det"].cpu().numpy())
        buf["risk_pred"].append(out["risk_mu"].cpu().numpy())
        buf["risk_logv"].append(out["risk_logv"].cpu().numpy())
        buf["risk_true"].append(batch["y_risk"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in buf.items()}


def report_classification(y_true, y_pred, y_prob, name, name_map=None, multiclass=True):
    avg = "macro" if multiclass else "binary"
    print(f"\n=== {name} ===")
    print(f"Accuracy       : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced Acc.  : {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision({avg}): {precision_score(y_true, y_pred, average=avg, zero_division=0):.4f}")
    print(f"Recall   ({avg}): {recall_score(y_true, y_pred, average=avg, zero_division=0):.4f}")
    print(f"F1       ({avg}): {f1_score(y_true, y_pred, average=avg, zero_division=0):.4f}")
    try:
        if multiclass and y_prob.shape[1] > 2:
            print(f"ROC-AUC (OvR)  : {roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'):.4f}")
        else:
            p = y_prob[:, 1] if (y_prob.ndim == 2 and y_prob.shape[1] == 2) else y_prob
            print(f"ROC-AUC        : {roc_auc_score(y_true, p):.4f}")
            print(f"PR-AUC         : {average_precision_score(y_true, p):.4f}")
            print(f"Brier          : {brier_score_loss(y_true, p):.4f}")
    except Exception as e:
        print("AUC skipped:", e)

    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    pretty = [name_map.get(int(l), str(l)) for l in labels] if name_map else [str(l) for l in labels]
    print("Confusion matrix (rows=true, cols=pred):")
    print("Labels:", pretty)
    print(cm)