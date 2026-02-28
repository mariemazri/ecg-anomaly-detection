"""
data_utils.py — Dataset generation & preprocessing for ECG anomaly detection.
Simulates a realistic ECG-like time series with injected anomalies.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


RANDOM_SEED = 42


def generate_ecg_dataset(
    n_normal: int = 4000,
    n_anomaly: int = 400,
    seq_len: int = 140,
    noise_std: float = 0.05,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic ECG-like multivariate time series dataset.

    Normal samples follow a realistic QRS complex pattern (P wave + QRS + T wave).
    Anomalies are injected as: spikes, flat-line segments, or amplitude shifts.

    Parameters
    ----------
    n_normal   : number of normal heartbeat windows
    n_anomaly  : number of anomalous windows
    seq_len    : number of time steps per window
    noise_std  : Gaussian noise level
    random_state : RNG seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (n_samples, seq_len)
    y : np.ndarray of shape (n_samples,)  — 0=normal, 1=anomaly
    """
    rng = np.random.RandomState(random_state)
    t = np.linspace(0, 2 * np.pi, seq_len)

    def _normal_beat(t: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Simulate one normal ECG beat: P + QRS + T wave."""
        # P wave
        p = 0.15 * np.exp(-0.5 * ((t - 0.8) / 0.15) ** 2)
        # QRS complex
        q = -0.1 * np.exp(-0.5 * ((t - 1.1) / 0.04) ** 2)
        r = 1.0  * np.exp(-0.5 * ((t - 1.2) / 0.06) ** 2)
        s = -0.15 * np.exp(-0.5 * ((t - 1.3) / 0.04) ** 2)
        # T wave
        t_wave = 0.3 * np.exp(-0.5 * ((t - 1.7) / 0.2) ** 2)
        # Slight random amplitude / phase shift per beat
        amplitude = rng.uniform(0.85, 1.15)
        phase = rng.uniform(-0.05, 0.05)
        beat = amplitude * (p + q + r + s + t_wave)
        beat += rng.normal(0, noise_std, size=seq_len)
        return np.roll(beat, int(phase * seq_len))

    # --- Normal samples ---
    X_normal = np.array([_normal_beat(t, rng) for _ in range(n_normal)])

    # --- Anomalous samples ---
    anomaly_types = ["spike", "flatline", "amplitude_shift"]
    X_anomaly_list = []
    for _ in range(n_anomaly):
        base = _normal_beat(t, rng)
        kind = rng.choice(anomaly_types)
        if kind == "spike":
            idx = rng.randint(0, seq_len)
            base[idx] += rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
        elif kind == "flatline":
            start = rng.randint(10, seq_len - 30)
            length = rng.randint(15, 30)
            base[start : start + length] = rng.uniform(-0.05, 0.05)
        else:  # amplitude_shift
            base *= rng.choice([0.2, 2.8])
        X_anomaly_list.append(base)
    X_anomaly = np.array(X_anomaly_list)

    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = RANDOM_SEED,
) -> Tuple:
    """
    Stratified chronological split: train / val / test.
    Stratification ensures class balance across splits.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )
    adjusted_val = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def clean_and_scale(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Data cleaning pipeline:
      1. Clip extreme values beyond ±5 std (robust to spike artifacts).
      2. Z-score normalisation fitted on train set only (no data leakage).

    Returns scaled arrays and the fitted scaler for inference.
    """
    # Step 1 — clip outliers (computed on train)
    mu = X_train.mean()
    sigma = X_train.std()
    X_train = np.clip(X_train, mu - 5 * sigma, mu + 5 * sigma)
    X_val   = np.clip(X_val,   mu - 5 * sigma, mu + 5 * sigma)
    X_test  = np.clip(X_test,  mu - 5 * sigma, mu + 5 * sigma)

    # Step 2 — z-score normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler
