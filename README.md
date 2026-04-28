# IoT Device Identification and Anomaly Detection (DIAD)

This repository contains the machine learning code accompanying the bachelor's thesis *Behavioural Analysis of IoT Network Traffic for Device Identification and Anomaly Detection* (Colin Duignan, Noroff University College, 2026). The work develops a Device Identification and Anomaly Detection (DIAD) pipeline for IoT network traffic, using Random Forest classifiers across five progressive models.

## Pipeline

The DIAD framework is a two-stage hybrid pipeline combining packet-level device identification with flow-level anomaly detection. Static identifiers (MAC, IP, ports, vendor strings) are stripped during preprocessing so classification relies on communication behaviour alone.

coldui01174_BPR_Artifact.png

## Models

| Script | Purpose |
|---|---|
| `DI_Model_1.py` | Device type classification (7 categories) |
| `DI_Model_2.py` | Per-category instance identification |
| `DI_Model_3.py` | Global instance identification (50 devices) |
| `AD_Model_1.py` | Binary anomaly detection (benign vs anomalous) |
| `AD_Model_2.py` | Multi-class attack classification |

The anomaly detection models (AD_Model_1 and AD_Model_2) use a compact 9-feature flow-level representation derived from stream, channel, and jitter statistics (mean, variance, count) summarized across five aggregation windows (1, 5, 10, 30, 60 seconds).

## Dataset

This study uses the **CIC IoT-DIAD 2024** dataset from the Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/). The dataset is not included in this repository due to size and licensing.

Download it from the source and preprocess into the input CSVs expected by each script (see the docstring at the top of each script for the expected filename).

## Requirements

- Python 3
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

Install with:

​```bash
pip install scikit-learn pandas numpy matplotlib joblib
​```

## Run

Each script is self-contained. From the repo root, with the input CSVs in place:

​```bash
python3 DI_Model_1.py
python3 DI_Model_2.py
python3 DI_Model_3.py
python3 AD_Model_1.py
python3 AD_Model_2.py
​```

## Reproducibility

All scripts use a fixed random seed of 42 for the train/test split, Random Forest bootstrap sampling, and per-device row sampling. Reported results are reproducible from the same input data.

## Reference

For full methodology, results, and discussion, see the accompanying thesis.
