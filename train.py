"""
train.py — Entry point for training the anomaly detection model.

Usage:
    python train.py --n_estimators 200 --contamination 0.091 --max_features 0.8

Environment:
    Python 3.10+, scikit-learn>=1.3, joblib, numpy, pandas
    requirements.txt at project root
    No GPU required.
"""

import argparse
import random
import numpy as np
from pathlib import Path

# ── Reproducibility: set all seeds before any imports that use RNG ──────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Note: sklearn uses numpy's global RNG; IsolationForest also takes random_state=SEED
# Remaining non-determinism: joblib parallel execution order (n_jobs=-1)
# can vary across runs due to OS scheduling. To eliminate: set n_jobs=1.

from src.data_utils import (
    generate_ecg_dataset,
    train_val_test_split,
    clean_and_scale,
)
from src.model import build_model, train, predict, evaluate, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ECG Anomaly Detector")
    p.add_argument("--n_estimators",  type=int,   default=200,   help="Number of isolation trees")
    p.add_argument("--contamination", type=float, default=0.091, help="Expected anomaly ratio")
    p.add_argument("--max_features",  type=float, default=0.8,   help="Feature subsampling ratio")
    p.add_argument("--n_normal",      type=int,   default=4000,  help="Normal samples to generate")
    p.add_argument("--n_anomaly",     type=int,   default=400,   help="Anomalous samples to generate")
    p.add_argument("--model_dir",     type=str,   default="models", help="Where to save checkpoint")
    p.add_argument("--log_dir",       type=str,   default="logs",   help="Where to save metric logs")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Data generation ─────────────────────────────────────────────────
    print("\n[1/4] Generating synthetic ECG dataset …")
    X, y = generate_ecg_dataset(
        n_normal=args.n_normal,
        n_anomaly=args.n_anomaly,
        random_state=SEED,
    )
    print(f"      Dataset: {X.shape[0]} samples × {X.shape[1]} features | "
          f"Anomaly ratio: {y.mean():.2%}")

    # ── 2. Split ───────────────────────────────────────────────────────────
    print("[2/4] Splitting into train / val / test (70 / 15 / 15, stratified) …")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, random_state=SEED
    )

    # ── 3. Preprocessing ───────────────────────────────────────────────────
    print("[3/4] Cleaning and scaling (clip ±5σ → z-score on train set only) …")
    X_train, X_val, X_test, scaler = clean_and_scale(X_train, X_val, X_test)

    # ── 4. Train & evaluate ────────────────────────────────────────────────
    print("[4/4] Training IsolationForest …")
    model = build_model(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_features=args.max_features,
        random_state=SEED,
    )
    model = train(model, X_train)

    for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred, scores = predict(model, Xs)
        evaluate(y_pred=y_pred, y_true=ys, scores=scores,
                 split_name=split, log_dir=Path(args.log_dir))

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_name = (f"iforest_n{args.n_estimators}"
                 f"_c{args.contamination}"
                 f"_f{args.max_features}.pkl")
    save_model(model, Path(args.model_dir) / ckpt_name)
    print(f"\nCheckpoint saved: {args.model_dir}/{ckpt_name}")


if __name__ == "__main__":
    main()
