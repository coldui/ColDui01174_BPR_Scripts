#!/usr/bin/env python3
"""
Train_Device_Model_Vs_All.py
DI Model 3 — Global Instance-Level Device Identification (Multiclass)

Goal: "Which exact device is this?" across all device instances simultaneously.
Label: src_mac (50 device classes)
Input: DI_Benign.csv

Notes:
  - This is the hardest of the three DI models: 50-way classification,
    high feature count, no category boundaries. Memory-conscious settings
    are required (depth cap, leaf-size floor, per-tree subsampling).
  - All hyperparameters are exposed as CLI flags so the exact run can be
    documented in the thesis appendix.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# --- Config ---
# Identifiers and IP-derived columns are dropped to enforce behavioural-only
# classification. Same set as Models 1 and 2.
DROP_COLS = {
    "src_mac",          # device instance label
    "src_port",         # ephemeral, can leak per-flow identity
    "dst_port",         # ephemeral, can leak per-flow identity
    "most_freq_spot",   # non-numeric metadata
    "l3_ip_dst_count",  # IP-derived
}
DROP_PREFIXES = (
    "src_ip_",
    "src_ip_mac_",
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage columns, coerce remaining columns to numeric, fill NaN/inf."""
    keep = [
        c for c in df.columns
        if c not in DROP_COLS
        and not c.startswith(DROP_PREFIXES)
    ]
    X = df[keep].copy()

    # Convert anything object-dtype to numeric where possible; drop the rest.
    for c in X.select_dtypes(include="object").columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.select_dtypes(include=[np.number])

    # Inf values arise from divide-by-zero in rate features; treat as 0.
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X.astype("float32")  # halves memory vs default float64


def cap_rows_per_class(df: pd.DataFrame, label_col: str, max_per_class: int, seed: int) -> pd.DataFrame:
    """Cap rows per device to balance classes and prevent OOM."""
    if max_per_class <= 0:
        return df
    return (
        df.groupby(label_col, group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), max_per_class), random_state=seed))
          .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="DI_Benign.csv")
    parser.add_argument("--label", default="src_mac")
    parser.add_argument("--max-per-device", type=int, default=100000,
                        help="Cap rows per device (0 disables)")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42,
                        help="Reproducibility seed for split, sampling, and RF")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-samples", type=float, default=0.8,
                        help="RF bootstrap subsample fraction per tree (memory)")
    parser.add_argument("--n-jobs", type=int, default=2,
                        help="Parallel jobs (avoid -1 to limit RAM spikes)")
    args = parser.parse_args()

    # Suppress pandas FutureWarning from groupby.apply (cosmetic only)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(args.csv):
        print(f"[!] Missing file: {args.csv}")
        sys.exit(1)

    # Stage 1: Load dataset
    df = pd.read_csv(args.csv, low_memory=False)

    if args.label not in df.columns:
        print(f"[!] Label column '{args.label}' not found.")
        sys.exit(1)

    # Stage 2: Drop unlabelled rows, then cap per-device rows for OOM safety
    df = df.dropna(subset=[args.label])
    df = cap_rows_per_class(df, args.label, args.max_per_device, args.seed)

    # Stage 3: Encode device labels
    le = LabelEncoder()
    y = le.fit_transform(df[args.label].astype(str).values)

    # Stage 4: Build feature matrix
    X = build_features(df)

    # Stage 5: Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # Stage 6: Random Forest with memory-conscious settings
    # max_samples and capped depth/leaf size are the OOM-control levers.
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )

    # Stage 7: Train and evaluate
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("=== DI Model 3 — Global Device Identification (50 classes) ===")
    print(f"Features used : {X.shape[1]}")
    print(f"Devices       : {len(le.classes_)}")
    print(f"Accuracy      : {acc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

    # Stage 8: Confusion matrix and top-10 most-confused device pairs
    cm = confusion_matrix(y_test, y_pred)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    top_idx = np.argsort(cm_off.ravel())[::-1][:10]

    print("Top-10 confusions (true -> pred = count):")
    for idx in top_idx:
        count = cm_off.ravel()[idx]
        if count <= 0:
            break
        i, j = idx // cm.shape[1], idx % cm.shape[1]
        print(f"  {le.classes_[i]} -> {le.classes_[j]} = {count}")

    # Stage 9: Save model, encoder, feature list, and confusion matrix
    joblib.dump(rf, "models/device_rf_model.joblib")
    joblib.dump(le, "models/device_label_encoder.joblib")
    joblib.dump(list(X.columns), "models/device_feature_columns.joblib")
    np.save("models/device_confusion_matrix.npy", cm)


if __name__ == "__main__":
    main()
