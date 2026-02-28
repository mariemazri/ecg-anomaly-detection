# ECG Anomaly Detection with Isolation Forest

> **M1 AI — Paris-Saclay** | Machine Learning Project | October 2025

Unsupervised anomaly detection on synthetic ECG-like time series using **Isolation Forest**.  
Detects three types of cardiac anomalies: spike artefacts, flat-line segments, and amplitude shifts.

---

## Project Structure

```
anomaly_detection/
├── src/
│   ├── data_utils.py       # Dataset generation, split, preprocessing
│   └── model.py            # IsolationForest wrapper, evaluation
├── notebooks/
│   └── ECG_AnomalyDetection.ipynb   # Full experimental notebook
├── tests/
│   └── test_pipeline.py    # 14 unit & integration tests
├── models/                 # Saved checkpoints (.pkl)
├── logs/                   # Metric JSONs + figures
├── train.py                # CLI training entry point
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | Synthetic ECG time series (custom) |
| **Samples** | 4 400 (4 000 normal + 400 anomalous) |
| **Features** | 140 (time steps per window) |
| **Anomaly ratio** | 9.09% |
| **Source** | Generated via `src/data_utils.generate_ecg_dataset()` |
| **License** | MIT (original synthetic data) |
| **Split** | 70 / 15 / 15 — stratified on class label |
| **Cleaning** | Clip ±5σ → z-score (fit on train only) |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (default config)
python train.py --n_estimators 200 --contamination 0.091 --max_features 0.8

# Run all tests
python -m unittest tests/test_pipeline.py -v

# Open notebook
jupyter notebook notebooks/ECG_AnomalyDetection.ipynb
```

---

## Results

| Split | F1 (Anomaly) | AUROC | Avg Precision |
|-------|-------------|-------|---------------|
| Val   | see logs    | see logs | see logs   |
| Test  | see logs    | see logs | see logs   |

*Logs saved automatically to `logs/metrics_val.json` and `logs/metrics_test.json`.*

---

## Model Choice

**Isolation Forest** was chosen over two alternatives:

| Model | Reason rejected |
|-------|----------------|
| One-Class SVM | O(n²) kernel computation prohibitive at 4 400 × 140; no scaling |
| LSTM Autoencoder | Requires GPU, complex to tune, overkill for this scope |

**Key hyperparameter:** `n_estimators` searched over {50, 100, 200, 500}.  
200 trees gave best F1/val with no meaningful gain beyond that.

**Supervision signal:** Fully unsupervised. Labels are NOT used during training.  
The pretext task is measuring average path length in random isolation trees  
(short path = easy to isolate = anomaly).

---

## Loss Function

$$\mathcal{L}(x) = \mathbb{E}[h(x)] \quad \text{where } h(x) = \text{path length in isolation tree}$$

No explicit regularisation. Implicit regularisation via `max_features=0.8` (feature subsampling).

---

## Reproducibility

```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
# IsolationForest(random_state=42)
```

**Remaining non-determinism:** parallel tree construction with `n_jobs=-1`  
can vary in execution order across runs. Use `n_jobs=1` for strict reproducibility.

---

## Tests

14 tests in `tests/test_pipeline.py` covering:
- Output shapes and label validity
- Reproducibility with fixed seeds
- No data leakage (train ↔ test)
- Scaler fitted on train only
- Model output format and score distribution

---

## Responsible AI

- **Bias:** Synthetic data does not capture real ECG artifacts (motion, electrode noise, patient morphology). Model should be validated on real data (e.g., MIT-BIH Arrhythmia Database) before clinical use.
- **Limitation:** Amplitude-shift anomalies are hardest to detect (short-path heuristic fails when scale changes but pattern is preserved).

---

## License

MIT — compatible with all dependencies (scikit-learn BSD-3, numpy BSD-3, pandas BSD-3).
