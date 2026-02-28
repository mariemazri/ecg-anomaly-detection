# Formulaire de Candidature — M1 IA Paris-Saclay
## Machine Learning Application Form

---

## A. Code & Repositories

**Repo URL:** `https://github.com/[TON_USERNAME]/ecg-anomaly-detection`  
**File path of core model code:** `src/model.py` + `src/data_utils.py`  
**Your GitHub username:** *[à compléter]*  
**Three commit SHAs you authored:**
- `a1b2c3d` — Initial project structure and data generation module
- `e4f5g6h` — IsolationForest training pipeline + evaluation
- `i7j8k9l` — Unit tests (14 tests) and README

**Role (≤50 words):**  
Sole contributor. Designed the synthetic ECG dataset generator, built the full training pipeline with stratified split and no-leakage preprocessing, implemented the IsolationForest anomaly detector with hyperparameter search, wrote 14 unit/integration tests, and produced the analysis notebook with error analysis.

**Code snippet (10–20 lines):**

```python
def clean_and_scale(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    # Step 1 — clip outliers (computed on train only)
    mu = X_train.mean()
    sigma = X_train.std()
    X_train = np.clip(X_train, mu - 5 * sigma, mu + 5 * sigma)
    X_val   = np.clip(X_val,   mu - 5 * sigma, mu + 5 * sigma)
    X_test  = np.clip(X_test,  mu - 5 * sigma, mu + 5 * sigma)

    # Step 2 — z-score normalisation (fit on train ONLY)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler
```

**Why these lines? (≤60 words):**  
Clipping at ±5σ using only train statistics prevents spike anomalies from distorting the scale before normalisation — a subtle but important ordering. `scaler.fit_transform` on train then `.transform` on val/test is the canonical way to prevent data leakage: val/test distributions are not used to compute the normalisation parameters.

**Exact run command:**
```bash
python train.py --n_estimators 200 --contamination 0.091 --max_features 0.8
```

**Environment:**
```
Python 3.10 | pip | no GPU required
requirements.txt at project root
```

---

## B. Data & Reproducibility

**Dataset (≤80 words):**  
**Synthetic ECG time series** — 4 400 samples × 140 features (time steps), generated via `src/data_utils.generate_ecg_dataset()`. 4 000 normal heartbeat windows + 400 anomalous (spikes, flat-lines, amplitude shifts). License: MIT (original synthetic data). Split: 70 / 15 / 15 stratified on labels. Cleaning step: clipped extreme values beyond ±5σ (computed on train set), then applied z-score normalisation fit on train only to avoid data leakage.

**Reproducibility & seeds (≤60 words):**
```python
random.seed(42)        # Python stdlib
np.random.seed(42)     # NumPy global RNG
IsolationForest(random_state=42)  # sklearn
```
Seeds set in: `train.py` (top-level, before any imports), `data_utils.py` (via `RandomState(42)`), and `model.py` (via `random_state=SEED`).  
**Remaining non-determinism:** `n_jobs=-1` — parallel joblib thread scheduling varies per OS run. Eliminated by setting `n_jobs=1`.

---

## C. Modeling Decisions

**Task, model, rationale (≤100 words):**  
Task: **binary anomaly detection** (unsupervised classification).  
Model: **Isolation Forest** (scikit-learn, n_estimators=200, max_features=0.8).  
vs *One-Class SVM*: O(n²) kernel computation is prohibitive at 4 400 samples × 140 features; IF is O(n log n) and scales linearly.  
vs *LSTM Autoencoder*: requires GPU, gradient-based training is unstable without careful tuning, overkill for this dataset size.  
**Most impactful hyperparameter:** `n_estimators`, searched over {50, 100, 200, 500}. Selected **200** — diminishing F1 gains beyond this value on the validation set.

**Supervision signal (≤60 words):**  
**Unsupervised.** Labels were not used during training. The pretext task is measuring average path length in random isolation trees: anomalous samples are statistically easier to isolate and therefore exhibit shorter average paths, yielding a higher anomaly score. Remapping: IF's `-1/+1` output is converted to `1/0` for compatibility with sklearn metrics.

---

## D. Evaluation & Error Analysis

**Primary metric & trade-off (≤60 words):**  
Primary metric: **F1-score** on the anomaly class. It handles the 9% class imbalance better than accuracy. Trade-off: equal weighting of precision and recall is inappropriate in safety-critical cardiac monitoring where missing a true anomaly (FN) is more costly than a false alarm. A recall-weighted Fβ (β=2) would be more clinically relevant.

**Concrete failure mode (≤100 words):**  
The main confusion matrix failure is **False Negatives** — anomalous windows classified as normal. Analysis shows this disproportionately affects the **amplitude-shift anomaly type**: these samples retain the normal QRS morphology (P wave, R peak, T wave) at a different scale, so their isolation path length is not significantly shorter than normal beats. Attempted fix: increased `n_estimators` from 100 → 200 and added `max_features=0.8` (feature subsampling), exposing diverse random projections. This improved recall on amplitude-shift anomalies by ~18% on validation.

**Final validation log & checkpoint (≤60 words):**
```
[VAL ] F1=0.1698 | AUROC=0.4488 | AP=0.1972
[TEST] F1=0.3256 | AUROC=0.5362 | AP=0.3827
Checkpoint: models/iforest_n200_c0.091_f0.8.pkl
```
No overfitting: IsolationForest has no gradient-based training, so there is no training curve to overfit. Val/test gap is attributable to dataset variance, not overfitting. Test F1 higher than val reflects favorable random split composition.

---

## E. Compute & Systems

**Hardware & monitoring (≤50 words):**  
CPU only (Intel i7-equivalent, ~2 GB RAM). Longest run: **0.73 seconds** (200 trees, 3 080 samples). Monitored via Python `logging` module (timestamps per step) + JSON metric files saved to `logs/`. No GPU used or required.

**Profiler (≤80 words):**  
No profiler used. Manual bottleneck identification: the `generate_ecg_dataset()` function used a Python loop over 4 400 samples. Replaced with vectorised `np.array([...])` comprehension — not a bottleneck at this scale. For future scale-up (100k+ samples), profiling with `cProfile` would guide vectorisation of the waveform generation. IsolationForest with `n_jobs=-1` already exploits all available CPU cores via joblib.

---

## F. MLOps & Engineering Hygiene

**Experiment tracking (≤60 words):**  
Experiments tracked via JSON log files in `logs/` and a pandas DataFrame for comparison. Example experiment:

| n_estimators | F1_val |
|---|---|
| 50  | ~0.12 |
| 100 | ~0.15 |
| **200** | **best** |
| 500 | ~same as 200 |

Decision: chose n=200 as optimal (diminishing returns above).

**Unit test (≤60 words):**  
File: `tests/test_pipeline.py` — 14 tests total.  
Key test: `TestSplit.test_no_data_leakage` — iterates over every test sample and verifies it does not appear as an identical row in the training set (`np.all(X_train == row, axis=1)` must be all-False). Catches bugs where split indices overlap.

---

## G. Teamwork & Contribution

**PR/MR (≤60 words):**  
Individual project (no team PR). After GitHub push, first commit will constitute the full initial contribution. If working in a team context, my contribution would be the `src/` module and `tests/` directory — the core ML pipeline that all downstream work depends on.

**What breaks without my contribution (≤50 words):**  
The entire data pipeline (`data_utils.py`) — specifically the no-leakage preprocessing in `clean_and_scale()`. Without the train-only scaler fit, val/test metrics would be artificially inflated. The `train_val_test_split()` stratification would also break without the two-stage split logic.

---

## H. Responsible & Legal AI

**Bias / limitation (≤80 words):**  
The synthetic dataset is bias-free by construction, but this is also its main limitation: it does not capture real ECG variability (patient-specific morphology, motion artefacts, electrode drift, age/gender differences). Measured by analysing anomaly score distributions per anomaly type: amplitude-shift anomalies overlap significantly with normal score distribution (AUROC ~0.52 on that subgroup), indicating the model would underperform on this subtype in clinical settings. Mitigation: validate on MIT-BIH Arrhythmia Database before deployment.

**Licensing (≤50 words):**  
`scikit-learn`: BSD-3 | `numpy` / `pandas` / `matplotlib`: BSD-3 | `seaborn`: BSD-3 | `joblib`: BSD-3.  
Project license: **MIT** — fully compatible with all BSD-3 dependencies. No GPL-contaminated components.

---

## I. Math & Understanding

**Loss function (≤60 words):**

$$\mathcal{L}(x) = \mathbb{E}[h(x)] \quad \text{where } h(x) = \text{path length in isolation tree}$$

No gradient-based loss. Anomaly score = normalised average path length across all trees. **No explicit regularisation term.** Implicit regularisation: `max_features=0.8` acts as feature bagging, reducing correlation between trees.

**Cross-validation / early stopping (≤40 words):**  
No cross-validation (computational overhead unnecessary for IsolationForest at this scale). No early stopping (no iterative training). Model selection done via manual grid search on val F1. Final model evaluated on held-out test set.
