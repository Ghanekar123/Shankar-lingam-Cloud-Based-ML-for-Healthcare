"""SHAP explainability + attention inspection."""

import numpy as np
import torch

from config import DEVICE


def run_shap_analysis(model, train_ds, test_ds, top_k=15, bg_size=100, sample_size=30):
    try:
        import shap

        feature_names = train_ds.feature_cols

        def predict_fn(X_np):
            model.eval()
            with torch.no_grad():
                X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
                ehr    = X[:, train_ds.ehr_idx]
                iot    = X[:, train_ds.iot_idx]
                physio = X[:, train_ds.physio_idx]
                lab    = X[:, train_ds.lab_idx]
                out = model(ehr, iot, physio, lab)
                return out["risk_mu"].cpu().numpy()

        bg = train_ds.X[:bg_size].numpy()
        smp = test_ds.X[:sample_size].numpy()
        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_vals = explainer.shap_values(smp, nsamples=60)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_k]

        print(f"\n=== Top-{top_k} influential features (SHAP on risk_score) ===")
        for i in top_idx:
            print(f"  {feature_names[i]:<25s}  {mean_abs[i]:.5f}")
        return shap_vals
    except Exception as e:
        print("\nSHAP skipped:", e)
        return None


def inspect_attention(model, test_loader):
    """Print sample of adaptive fusion alpha values."""
    model.eval()
    with torch.no_grad():
        b = next(iter(test_loader))
        for k in b:
            b[k] = b[k].to(DEVICE)
        o = model(b["ehr"], b["iot"], b["physio"], b["lab"])
        print("\nAdaptive fusion alpha (first 5):", o["alpha"][:5].cpu().numpy().ravel())