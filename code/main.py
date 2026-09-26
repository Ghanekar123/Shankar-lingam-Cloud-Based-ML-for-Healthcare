"""End-to-end SAM-CRN pipeline: load -> split -> train -> evaluate -> explain."""

import os
import json
import math
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    SEED, CSV_PATH, ARTIFACT_DIR, DEVICE,
    EHR_COLS, IOT_COLS, PHYSIO_COLS, LAB_COLS,
    ID_COLS, TARGET_COLS, FEATURE_COLS,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, HIDDEN_DIM,
)
from preprocessing import preprocess_dataframe, encode_targets
from split import split_and_report
from dataset import MultimodalHealthcareDataset
from model import SAM_CRN
from losses import MultitaskLoss
from trainer import train_model
from metrics import evaluate, report_classification
from anomaly import run_anomaly_detection
from explainability import run_shap_analysis, inspect_attention


def main():
    # ---------------- Reproducibility ----------------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print("Device:", DEVICE)

    # ---------------- Load ----------------
    df = pd.read_csv(CSV_PATH)
    print("Raw shape:", df.shape)
    print(df.head())

    # ---------------- Preprocess ----------------
    df_proc, scaler = preprocess_dataframe(df)
    print("Processed shape:", df_proc.shape)

    df_proc, le_diag, le_sev, target_name_maps, num_diag, num_sev = encode_targets(df_proc)

    print("\nClass name mappings:")
    for tgt, m in target_name_maps.items():
        print(f"  {tgt}: {m}")

    # ---------------- Split ----------------
    train_df, val_df, test_df = split_and_report(df_proc, target_name_maps, "diagnosis_class")

    # ---------------- Datasets ----------------
    train_ds = MultimodalHealthcareDataset(train_df, FEATURE_COLS, EHR_COLS, IOT_COLS, PHYSIO_COLS, LAB_COLS)
    val_ds   = MultimodalHealthcareDataset(val_df,   FEATURE_COLS, EHR_COLS, IOT_COLS, PHYSIO_COLS, LAB_COLS)
    test_ds  = MultimodalHealthcareDataset(test_df,  FEATURE_COLS, EHR_COLS, IOT_COLS, PHYSIO_COLS, LAB_COLS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    input_dim = len(train_ds.feature_cols)
    print("\nInput feature dim:", input_dim)

    # ---------------- Model ----------------
    model = SAM_CRN(
        in_dim=input_dim,
        ehr_idx=train_ds.ehr_idx, iot_idx=train_ds.iot_idx,
        physio_idx=train_ds.physio_idx, lab_idx=train_ds.lab_idx,
        n_diag=num_diag, n_sev=num_sev, hidden=HIDDEN_DIM,
    ).to(DEVICE)

    criterion = MultitaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------------- Train ----------------
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)

    # ---------------- Evaluate ----------------
    print("\n" + "=" * 78)
    print("TEST-SET EVALUATION (with class names)")
    print("=" * 78)
    res = evaluate(model, test_loader)

    report_classification(res["diag_true"], res["diag_pred"], res["diag_prob"],
                          "Diagnosis (multiclass)",
                          name_map=target_name_maps["diagnosis_class"])

    report_classification(res["sev_true"], res["sev_pred"], res["sev_prob"],
                          "Severity (multiclass)",
                          name_map=target_name_maps["severity_class"])

    report_classification(res["det_true"], (res["det_prob"] > 0.5).astype(int),
                          res["det_prob"], "Deterioration (binary)",
                          name_map=target_name_maps["deterioration_label"],
                          multiclass=False)

    print("\n=== Risk Regression (risk_score) ===")
    print(f"MAE : {mean_absolute_error(res['risk_true'], res['risk_pred']):.4f}")
    print(f"RMSE: {math.sqrt(mean_squared_error(res['risk_true'], res['risk_pred'])):.4f}")
    print(f"Mean predictive std (uncertainty): {np.exp(0.5 * res['risk_logv']).mean():.4f}")

    # ---------------- Explainability ----------------
    run_shap_analysis(model, train_ds, test_ds, top_k=15)
    inspect_attention(model, test_loader)

    # ---------------- Anomaly Detection ----------------
    run_anomaly_detection(model, train_loader, test_loader, contamination=0.05)

    # ---------------- Save Artifacts ----------------
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "sam_crn.pt"))
    with open(os.path.join(ARTIFACT_DIR, "config.json"), "w") as f:
        json.dump({
            "feature_cols": train_ds.feature_cols,
            "ehr_idx": train_ds.ehr_idx, "iot_idx": train_ds.iot_idx,
            "physio_idx": train_ds.physio_idx, "lab_idx": train_ds.lab_idx,
            "n_diag": num_diag, "n_sev": num_sev,
            "diag_classes": list(le_diag.classes_),
            "sev_classes":  list(le_sev.classes_),
            "target_name_maps": {k: {str(a): b for a, b in v.items()}
                                 for k, v in target_name_maps.items()},
        }, f, indent=2)
    train_df.to_csv(os.path.join(ARTIFACT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(ARTIFACT_DIR, "val.csv"),     index=False)
    test_df.to_csv(os.path.join(ARTIFACT_DIR, "test.csv"),   index=False)

    print("\nAll artifacts saved to ./artifacts/")
    print("DONE.")


if __name__ == "__main__":
    main()
    