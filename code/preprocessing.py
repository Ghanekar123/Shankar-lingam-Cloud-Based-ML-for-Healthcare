"""Data cleaning, encoding, and target preparation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import CAT_COLS, IOT_COLS, FEATURE_COLS


def preprocess_dataframe(df: pd.DataFrame):
    """Clean outliers, encode categoricals, scale numerics."""
    df = df.copy()

    # Outlier clipping on IoT signals (IQR)
    for c in [c for c in IOT_COLS if c in df.columns]:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan

    # Label-encode categoricals
    for c in CAT_COLS:
        if c not in df.columns:
            continue
        df[c] = df[c].astype(str)
        df[c] = LabelEncoder().fit_transform(df[c])

    # Median imputation + scaling
    num_cols = [c for c in FEATURE_COLS
                if c in df.columns and df[c].dtype != object]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


def encode_targets(df: pd.DataFrame):
    """Encode classification targets, return encoders + name maps."""
    le_diag = LabelEncoder()
    df["diagnosis_class"] = le_diag.fit_transform(df["diagnosis_class"].astype(str))

    le_sev = LabelEncoder()
    df["severity_class"] = le_sev.fit_transform(df["severity_class"].astype(str))

    det_name_map = {0: "No Deterioration", 1: "Deterioration"}

    df["risk_score"] = df["risk_score"].clip(0, 1)
    df["deterioration_label"] = df["deterioration_label"].astype(int)

    target_name_maps = {
        "diagnosis_class":     {i: n for i, n in enumerate(le_diag.classes_)},
        "severity_class":      {i: n for i, n in enumerate(le_sev.classes_)},
        "deterioration_label": det_name_map,
    }

    num_diag = len(le_diag.classes_)
    num_sev = len(le_sev.classes_)
    return df, le_diag, le_sev, target_name_maps, num_diag, num_sev