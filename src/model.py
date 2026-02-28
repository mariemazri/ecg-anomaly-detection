"""
model.py — Isolation Forest anomaly detector with full evaluation pipeline.

Design choice: Isolation Forest over:
  - One-Class SVM  → doesn't scale to 4400 samples × 140 features
  - LSTM Autoencoder → requires GPU, overkill for M1 coursework scope
"""

import numpy as np
import logging
import time
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
)
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

RANDOM_SEED = 42


def build_model(
    n_estimators: int = 200,
    max_samples: str = "auto",
    contamination: float = 0.091,   # 400 / 4400 ≈ 9.1 %
    max_features: float = 0.8,
    random_state: int = RANDOM_SEED,
) -> IsolationForest:
    """
    Instantiate an Isolation Forest.

    Key hyperparameter: n_estimators=200 (searched 50–500).
    More trees reduce variance of anomaly scores; beyond 200 gain is marginal.
    contamination reflects the true anomaly ratio in the dataset.
    max_features=0.8 introduces feature bagging, improving generalisation.
    """
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )


def train(
    model: IsolationForest,
    X_train: np.ndarray,
) -> IsolationForest:
    """
    Fit the model on training data only (unsupervised: labels not used).
    Supervision signal: unsupervised / self-supervised via path length
    in random isolation trees as proxy for anomaly score.
    """
    logger.info("Training IsolationForest on %d samples × %d features …",
                X_train.shape[0], X_train.shape[1])
    t0 = time.time()
    model.fit(X_train)
    elapsed = time.time() - t0
    logger.info("Training done in %.2f s", elapsed)
    return model


def predict(model: IsolationForest, X: np.ndarray) -> tuple:
    """
    Returns binary predictions and raw anomaly scores.
    IsolationForest.predict returns +1 (normal) / -1 (anomaly).
    We remap to 0/1 for standard sklearn metrics.
    """
    raw = model.predict(X)                  # +1 or -1
    y_pred = (raw == -1).astype(int)        # 0=normal, 1=anomaly
    scores = -model.score_samples(X)        # higher → more anomalous
    return y_pred, scores


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    split_name: str = "val",
    log_dir: Path = Path("logs"),
) -> dict:
    """
    Full evaluation suite:
      - F1 (primary metric, handles class imbalance)
      - AUROC, Average Precision
      - Confusion matrix
      - Classification report

    F1 trade-off: optimises recall on anomalies but may inflate false positives.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    report = classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"])

    metrics = {
        "split": split_name,
        "f1_anomaly": round(f1, 4),
        "auroc": round(auroc, 4),
        "avg_precision": round(ap, 4),
        "confusion_matrix": cm.tolist(),
    }

    log_line = (
        f"[{split_name.upper()}] F1={f1:.4f} | AUROC={auroc:.4f} | AP={ap:.4f}"
    )
    logger.info(log_line)
    logger.info("\n%s", report)

    # Save metrics to JSON log
    log_path = log_dir / f"metrics_{split_name}.json"
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def save_model(model: IsolationForest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)


def load_model(path: Path) -> IsolationForest:
    return joblib.load(path)
