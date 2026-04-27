# IoT Device Identification and Anomaly Detection (DIAD)

This repository contains the machine learning code accompanying the bachelor thesis BEHAVIOURAL ANALYSIS OF IOT NETWORK TRAFFIC FOR DEVICE
IDENTIFICATION AND ANOMALY DETECTION (Noroff University College, 2026). The work develops a Device Identification and Anomaly Detection (DIAD) pipeline for IoT network traffic, using Random Forest classifiers across five progressive models.

## Models

| Script | Purpose |

| `DI_Model_1.py` | Device type classification (7 categories) |
| `DI_Model_2.py` | Per-category instance identification |
| `DI_Model_3.py` | Global instance identification (50 devices) |
| `AD_Model_1.py` | Binary anomaly detection |
| `AD_Model_2.py` | Multi-class attack classification |

## Dataset

This study uses the **CIC IoT-DIAD 2024** dataset from the Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/). The dataset is not included in this repository due to size and licensing. 
Download it from the source and preprocess into the input CSVs expected by each script (see the docstring at the top of each script for the expected filename).

## Our VENV Setup

```bash
python3 -m venv sklearn-env
source sklearn-env/bin/activate
pip install any necessary packages
```

## Run

Each script is self-contained. From the repo root, with the input CSVs in place:

```bash

python3 DI_Model_1.py
python3 DI_Model_2.py
python3 DI_Model_3.py
python3 AD_Model_1.py
python3 AD_Model_2.py

```


## Reproducibility

All scripts use a fixed random seed of 42 for the train/test split, Random Forest bootstrap sampling, and per-device row sampling. Reported results are reproducible from the same input data.
