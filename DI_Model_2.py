#!/usr/bin/env python3
"""
Train_Device_Model2_PerCategory.py
DI Model 2 — Instance-Level Device Identification within each Device Type

Goal: "Which specific device is this?" — trained separately per category.
Label: src_mac (device instance)
Split: by device_type (category)
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
INFILE  = "DI_Benign_with_Types.csv"
DI_COL  = "src_mac"
CAT_COL = "device_type"

# Fixed seed for reproducibility: controls the train/test split and the
# Random Forest's bootstrap sampling so results are repeatable across runs.
RANDOM_STATE = 42

# Static identifiers, IP-derived columns, and the category column itself
# are dropped. Category is the splitter (constant within each partition),
# not a feature, so it must be excluded.
DROP_COLS = {
    "src_mac", "dst_mac",
    "src_ip", "dst_ip",
    "src_port", "dst_port",
    "eth_src_oui", "eth_dst_oui",
    "http_host", "http_uri",
    "tls_server", "User_Agent",
    "most_freq_spot",
    "l3_ip_dst_count",
    "device_type",
}
DROP_PREFIXES = (
    "src_ip_",
    "dst_ip_",
    "src_ip_mac_",
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage columns and any non-numeric columns."""
    keep = [
        c for c in df.columns
        if c not in DROP_COLS
        and not c.startswith(DROP_PREFIXES)
    ]
    return df[keep].select_dtypes(include=[np.number])


def run_category(df_cat: pd.DataFrame, category: str):
    """Train and evaluate one Random Forest for a single device category."""

    # Stage 1: Skip categories with fewer than 2 devices (classification undefined)
    if df_cat[DI_COL].nunique() < 2:
        print(f"--- {category}: skipped (only {df_cat[DI_COL].nunique()} device) ---")
        return

    # Stage 2: Build features and encode device-instance labels
    X = build_features(df_cat)
    le = LabelEncoder()
    y = le.fit_transform(df_cat[DI_COL].astype(str))

    # Stage 3: Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Stage 4: Build pipeline (median imputation + Random Forest)
    # balanced_subsample re-weights per bootstrap sample, more robust than
    # 'balanced' when per-category class distributions are highly skewed.
    clf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        )),
    ])

    # Stage 5: Train and evaluate
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc      = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"=== Category: {category} ({df_cat[DI_COL].nunique()} devices) ===")
    print(f"Features used : {X.shape[1]}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print()
    print(classification_report(y_test, preds, target_names=le.classes_, digits=4))

    # Stage 6: Save per-category artefacts
    safe_name = category.replace(" ", "_").lower()
    joblib.dump(clf, f"models/model2_{safe_name}.joblib")
    joblib.dump(le,  f"models/model2_{safe_name}_encoder.joblib")
    joblib.dump(list(X.columns), f"models/model2_{safe_name}_features.joblib")


def main():
    os.makedirs("models", exist_ok=True)

    # Stage 1: Load dataset
    df = pd.read_csv(INFILE, low_memory=False)

    # Stage 2: Loop over each device category and train a separate model
    for cat in sorted(df[CAT_COL].unique()):
        df_cat = df[df[CAT_COL] == cat].copy()
        run_category(df_cat, cat)


if __name__ == "__main__":
    main()
