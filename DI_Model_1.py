#!/usr/bin/env python3
"""
Train_Device_Type_Model.py
DI Model 1 — Device Type Classification (Multiclass) — Random Forest

Goal: "What type of device is this?"
Label: device_type (7 categories: Camera, Audio Device, Power Outlet,
       Home Automation, Lighting, Hub Device, Sensor)
Input: DI_Benign_with_Types.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# --- Config ---
INFILE    = "DI_Benign_with_Types.csv"
LABEL_COL = "device_type"

# Fixed seed for reproducibility: controls the train/test split and the
# Random Forest's bootstrap sampling so results are repeatable across runs.
RANDOM_STATE = 42

# Static identifiers and IP-derived columns are dropped to enforce
# behavioural-only classification (no leakage from MAC/IP/host strings).
DROP_COLS = {
    "src_mac", "dst_mac",
    "src_ip", "dst_ip",
    "src_port", "dst_port",
    "eth_src_oui", "eth_dst_oui",
    "http_host", "http_uri",
    "tls_server", "User_Agent",
    "most_freq_spot",
    "l3_ip_dst_count",
}
DROP_PREFIXES = (
    "src_ip_",
    "dst_ip_",
    "src_ip_mac_",
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop label, leakage columns, and any non-numeric columns."""
    keep = [
        c for c in df.columns
        if c != LABEL_COL
        and c not in DROP_COLS
        and not c.startswith(DROP_PREFIXES)
    ]
    return df[keep].select_dtypes(include=[np.number])


def main():
    os.makedirs("models", exist_ok=True)

    # Stage 1: Load dataset
    df = pd.read_csv(INFILE, low_memory=False)

    # Stage 2: Build feature matrix and encode labels
    X = build_features(df)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].astype(str))

    # Stage 3: Stratified 80/20 train-test split (preserves class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Stage 4: Build pipeline (median imputation + Random Forest)
    clf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    # Stage 5: Train
    clf.fit(X_train, y_train)

    # Stage 6: Evaluate on held-out test set
    preds = clf.predict(X_test)

    acc         = accuracy_score(y_test, preds)
    macro_f1    = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")

    print("=== DI Model 1 — Device Type Classification ===")
    print(f"Features used : {X.shape[1]}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")
    print()
    print(classification_report(y_test, preds, target_names=le.classes_, digits=4))

    # Stage 7: Save model, encoder, and feature list
    joblib.dump(clf, "models/device_type_model.joblib")
    joblib.dump(le,  "models/device_type_label_encoder.joblib")
    joblib.dump(list(X.columns), "models/device_type_feature_columns.joblib")


if __name__ == "__main__":
    main()
