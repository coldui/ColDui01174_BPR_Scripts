#!/usr/bin/env python3
"""
AD_Model1_Binary.py
AD Model 1 — Binary Anomaly Detection (per device, per attack) — Random Forest

Goal: "Is this traffic benign or attack?" trained separately for each
      (device, attack, window) combination.
Label: label (0 = benign, 1 = attack)
Input: per-pair balanced CSVs (e.g. arlo_q_mirai_balanced.csv)

Notes:
  - 3 devices × 6 attacks × 5 windows = up to 90 classifiers per run.
  - Each window uses only its own stream/channel/jitter feature triplet
    (9 features per window), so the per-window comparison is clean.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib


# --- Config ---
PROCESSED_DIR = Path("/home/kali/iot-di-ad/data/processed")
MODELS_DIR    = Path("/home/kali/iot-di-ad/models/ad_binary")

# Fixed seed for reproducibility: controls the train/test split and the
# Random Forest's bootstrap sampling so results are repeatable across runs.
RANDOM_STATE = 42

# Sliding windows (seconds) over which behavioural features are aggregated.
WINDOWS = [1, 5, 10, 30, 60]

# Camera devices in the AD case study (chosen as a focused subset of the dataset).
DEVICES = {
    "arlo_q":      "Arlo Q Indoor Camera",
    "nest_indoor": "Nest Indoor Camera",
    "netatmo":     "Netatmo Camera",
}

# Attack types covering five of the dataset's attack categories.
ATTACKS = ["mirai", "http", "ddos_tcp", "slowloris", "recon_osscan", "dns_spoofing"]


def select_window_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return only the stream/channel/jitter columns for the given window."""
    prefixes = (
        f"stream_{window}_",
        f"channel_{window}_",
        f"stream_jitter_{window}_",
    )
    keep = [c for c in df.columns if c != "label" and c.startswith(prefixes)]
    return df[keep].apply(pd.to_numeric, errors="coerce")


def run_pair(device_key: str, attack: str, df: pd.DataFrame) -> list:
    """Train one Random Forest per window for a single (device, attack) pair."""
    y = df["label"].astype(int)
    rows = []

    for window in WINDOWS:

        # Stage 1: Build feature matrix for this window
        X = select_window_features(df, window)
        if X.shape[1] == 0:
            continue

        # Stage 2: Stratified 80/20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        # Stage 3: Build pipeline (median imputation + Random Forest)
        clf = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

        # Stage 4: Train and evaluate
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        rows.append({
            "device":   device_key,
            "attack":   attack,
            "window":   window,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
            "support":  len(y_test),
            "errors":   int((y_test != y_pred).sum()),
        })

        # Stage 5: Save model artefact
        joblib.dump(clf, MODELS_DIR / f"{device_key}_{attack}_{window}s.joblib")

    return rows


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Stage 1: Loop over every (device, attack) pair and run all 5 windows
    for device_key, device_name in DEVICES.items():
        for attack in ATTACKS:
            infile = PROCESSED_DIR / f"{device_key}_{attack}_balanced.csv"
            if not infile.exists():
                print(f"--- {device_name} | {attack}: missing input, skipped ---")
                continue

            df = pd.read_csv(infile, low_memory=False)
            results = run_pair(device_key, attack, df)
            all_results.extend(results)

            # Stage 2: Print per-window summary for this pair
            print(f"=== {device_name} | {attack} ===")
            print(f"{'Window':>8} {'Accuracy':>10} {'Macro F1':>10} {'Support':>9} {'Errors':>8}")
            for r in results:
                print(f"{r['window']:>7}s {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} "
                      f"{r['support']:>9} {r['errors']:>8}")
            print()

    # Stage 3: Save full summary CSV across all pairs
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(PROCESSED_DIR / "ad_binary_summary.csv", index=False)


if __name__ == "__main__":
    main()
