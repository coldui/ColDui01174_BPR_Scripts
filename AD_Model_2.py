#!/usr/bin/env python3
"""
AD_Model2_Multiclass.py
AD Model 2 — Multi-Class Attack Classification — Random Forest

Goal: "If anomalous, which attack type is it?" — 7-class classification
      (benign + 6 attacks) trained per device across all 5 windows.
Label: attack_type (0 = benign, 1 = mirai, 2 = http, 3 = ddos_tcp,
       4 = slowloris, 5 = recon_osscan, 6 = dns_spoofing)
Input: per-pair balanced CSVs (e.g. arlo_q_mirai_balanced.csv)

Notes:
  - Benign rows are pooled from all 6 attack CSVs and deduplicated.
  - All 7 classes are balanced per device by undersampling to the smallest
    class size (566 for Arlo Q, 993 for Nest, 142 for Netatmo).
  - The best window per device is selected by accuracy and used for the
    confusion matrix and classification report output.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib


# --- Config ---
PROCESSED_DIR = Path("/home/kali/iot-di-ad/data/processed")
MODELS_DIR    = Path("/home/kali/iot-di-ad/models/ad_multiclass")
RESULTS_DIR   = Path("/home/kali/iot-di-ad/results/multiclass")

# Fixed seed for reproducibility: controls the train/test split, the
# Random Forest's bootstrap sampling, and the per-class balanced sampling.
RANDOM_STATE = 42

# Sliding windows (seconds) over which behavioural features are aggregated.
WINDOWS = [1, 5, 10, 30, 60]

# Camera devices in the AD case study.
DEVICES = {
    "arlo_q":      "Arlo Q Indoor Camera",
    "nest_indoor": "Nest Indoor Camera",
    "netatmo":     "Netatmo Camera",
}

# Attack types covering five of the dataset's attack categories.
# Order matters: ATTACKS[i-1] corresponds to LABEL_MAP[i].
ATTACKS = ["mirai", "http", "ddos_tcp", "slowloris", "recon_osscan", "dns_spoofing"]

LABEL_MAP = {
    0: "benign",
    1: "mirai",
    2: "http",
    3: "ddos_tcp",
    4: "slowloris",
    5: "recon_osscan",
    6: "dns_spoofing",
}


def select_window_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return only the 9 stream/channel/jitter features for the given window."""
    keep = [
        f"stream_{window}_count",
        f"stream_{window}_mean",
        f"stream_{window}_var",
        f"channel_{window}_count",
        f"channel_{window}_mean",
        f"channel_{window}_var",
        f"stream_jitter_{window}_sum",
        f"stream_jitter_{window}_mean",
        f"stream_jitter_{window}_var",
    ]
    X = df[[c for c in keep if c in df.columns]]
    return X.apply(pd.to_numeric, errors="coerce")


def save_confusion_matrix(cm, labels, device_key, window, outdir):
    """Render and save the confusion matrix for the device's best window."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, cmap="Greys")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {device_key} ({window}s window)")

    max_val = cm.max()
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > max_val / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"confusion_matrix_{device_key}_{window}s.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix: {outpath}")


def load_device_data(device_key: str):
    """Load benign + all attack rows for a device and balance to smallest class."""

    # Stage 1: Read every (device, attack) CSV, splitting benign from attack rows
    all_benign = []
    attack_dfs = {}

    for i, attack in enumerate(ATTACKS, start=1):
        infile = PROCESSED_DIR / f"{device_key}_{attack}_balanced.csv"
        if not infile.exists():
            print(f"  [!] Missing: {infile.name} — skipping")
            continue

        df = pd.read_csv(infile, low_memory=False)
        all_benign.append(df[df["label"] == 0].copy())

        attack_rows = df[df["label"] == 1].copy()
        attack_rows["attack_type"] = i
        attack_dfs[attack] = attack_rows

    if not attack_dfs:
        print(f"[!] No data found for {device_key}")
        return None

    # Stage 2: Pool benign rows across attack files, deduplicate, label as 0
    benign_combined = pd.concat(all_benign, ignore_index=True).drop_duplicates()
    benign_combined["attack_type"] = 0

    # Stage 3: Balance all 7 classes by undersampling to the smallest class
    all_dfs = [benign_combined] + list(attack_dfs.values())
    n = min(len(d) for d in all_dfs)
    print(f"  Balancing at n={n} per class")

    balanced = [d.sample(n=n, random_state=RANDOM_STATE) for d in all_dfs]
    combined = pd.concat(balanced, ignore_index=True)

    # Stage 4: Shuffle to remove ordering by class
    return combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def run_multiclass(device_key: str, device_name: str, df: pd.DataFrame):
    """Train and evaluate one multiclass RF per window for a single device."""
    y = df["attack_type"].astype(int)
    labels = [LABEL_MAP[i] for i in sorted(y.unique())]

    summary_rows = []
    best_acc, best_window, best_cm, best_report = 0, None, None, None

    for window in WINDOWS:

        # Stage 1: Build feature matrix for this window
        X = select_window_features(df, window)
        if X.shape[1] == 0:
            continue

        print(f"  Window {window}s | Features: {X.shape[1]}")

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

        acc      = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        cm       = confusion_matrix(y_test, y_pred)

        summary_rows.append({
            "device":   device_key,
            "window":   window,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "support":  len(y_test),
        })

        # Stage 5: Track the best-performing window for this device
        if acc > best_acc:
            best_acc, best_window, best_cm = acc, window, cm
            best_report = classification_report(
                y_test, y_pred, target_names=labels, digits=4
            )

        # Stage 6: Save model artefact
        joblib.dump(clf, MODELS_DIR / f"{device_key}_multiclass_{window}s.joblib")

    # Stage 7: Print per-window summary, with marker on best
    print(f"\n  {'Window':>8} {'Accuracy':>10} {'Macro F1':>10} {'Support':>9}")
    print("  " + "-" * 42)
    for r in summary_rows:
        marker = " <-- best" if r["window"] == best_window else ""
        print(f"  {r['window']:>7}s {r['accuracy']:>10.4f} "
              f"{r['macro_f1']:>10.4f} {r['support']:>9}{marker}")

    # Stage 8: Print classification report and save confusion matrix for best window
    print(f"\n  Best window: {best_window}s")
    print(f"  Classification Report ({best_window}s):")
    print(best_report)

    save_confusion_matrix(best_cm, labels, device_key, best_window, RESULTS_DIR)

    return summary_rows


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Stage 1: Loop over each camera device
    for device_key, device_name in DEVICES.items():
        print(f"\n{'='*60}")
        print(f"Device: {device_name}")

        # Stage 2: Load and balance data for this device
        df = load_device_data(device_key)
        if df is None:
            continue

        print(f"  Total rows: {len(df)}")
        print(f"  Class distribution:\n{df['attack_type'].value_counts().sort_index().to_string()}")

        # Stage 3: Train and evaluate across all windows
        results = run_multiclass(device_key, device_name, df)
        all_results.extend(results)

    # Stage 4: Save full summary CSV across all devices and windows
    summary_df = pd.DataFrame(all_results)
    out_csv = PROCESSED_DIR / "ad_multiclass_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[+] Full summary saved to {out_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
