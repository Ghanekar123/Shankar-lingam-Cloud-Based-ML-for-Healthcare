"""Global configuration for SAM-CRN."""

import torch

# ---------------- Reproducibility ----------------
SEED = 42

# ---------------- Paths ----------------
CSV_PATH = r"C:\Aakashimplement\ER298872_Dr. Shankar lingam_Nikhita\SAM_CRN_50000_patients_5M_observations.csv"
ARTIFACT_DIR = "artifacts"
CKPT_PATH = "best_sam_crn.pt"

# ---------------- Device ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Column groups ----------------
ID_COLS = ["patient_id", "observation_id", "timestamp"]

CAT_COLS = [
    "gender", "smoking_status", "previous_diagnosis",
    "symptoms", "medications", "clinical_event"
]

EHR_COLS = [
    "age", "gender", "height", "weight", "bmi",
    "diabetes", "hypertension", "cardiac_history", "respiratory_history",
    "smoking_status", "previous_diagnosis"
]

IOT_COLS = [
    "heart_rate", "spo2", "systolic_bp", "diastolic_bp",
    "respiratory_rate", "body_temperature",
    "activity_level", "sleep_duration", "signal_quality"
]

PHYSIO_COLS = ["ecg_features", "hrv_features", "time_since_admission"]

LAB_COLS = [
    "hemoglobin", "wbc", "platelets", "glucose", "creatinine",
    "sodium", "potassium", "crp"
]

TARGET_CLF = ["diagnosis_class", "severity_class", "deterioration_label"]
TARGET_REG = ["risk_score"]
TARGET_COLS = TARGET_CLF + TARGET_REG

# ---------------- Training ----------------
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64