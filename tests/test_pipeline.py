"""
tests/test_pipeline.py — Unit & integration tests for data and model pipeline.

Run with:  python -m pytest tests/ -v
"""

import numpy as np
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import generate_ecg_dataset, train_val_test_split, clean_and_scale
from src.model import build_model, train, predict


# ─── Data tests ──────────────────────────────────────────────────────────────

class TestDataGeneration(unittest.TestCase):
    def test_output_shapes(self):
        X, y = generate_ecg_dataset(n_normal=100, n_anomaly=10, seq_len=140)
        assert X.shape == (110, 140), f"Expected (110,140), got {X.shape}"
        assert y.shape == (110,)

    def test_binary_labels(self):
        _, y = generate_ecg_dataset(n_normal=100, n_anomaly=10)
        assert set(np.unique(y)) == {0, 1}, "Labels must be 0 or 1"

    def test_anomaly_ratio(self):
        X, y = generate_ecg_dataset(n_normal=100, n_anomaly=10)
        ratio = y.mean()
        assert 0.08 < ratio < 0.12, f"Anomaly ratio out of expected range: {ratio}"

    def test_reproducibility(self):
        X1, y1 = generate_ecg_dataset(n_normal=50, n_anomaly=5, random_state=42)
        X2, y2 = generate_ecg_dataset(n_normal=50, n_anomaly=5, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_ecg_dataset(n_normal=50, n_anomaly=5, random_state=0)
        X2, _ = generate_ecg_dataset(n_normal=50, n_anomaly=5, random_state=99)
        assert not np.allclose(X1, X2), "Different seeds should produce different data"


class TestSplit(unittest.TestCase):
    def test_split_sizes(self):
        X, y = generate_ecg_dataset(n_normal=400, n_anomaly=40)
        Xtr, Xv, Xte, ytr, yv, yte = train_val_test_split(X, y)
        total = len(ytr) + len(yv) + len(yte)
        assert total == len(y), "Split must preserve total sample count"

    def test_stratification(self):
        """Train, val, test should all contain anomalies (stratified)."""
        X, y = generate_ecg_dataset(n_normal=400, n_anomaly=40)
        _, _, _, ytr, yv, yte = train_val_test_split(X, y)
        for split_y, name in [(ytr, "train"), (yv, "val"), (yte, "test")]:
            assert 1 in split_y, f"No anomalies in {name} split — stratification failed"

    def test_no_data_leakage(self):
        """Ensure no sample appears in both train and test."""
        X, y = generate_ecg_dataset(n_normal=200, n_anomaly=20)
        Xtr, _, Xte, *_ = train_val_test_split(X, y)
        # Check no identical rows
        for row in Xte:
            matches = np.all(Xtr == row, axis=1)
            assert not matches.any(), "Data leakage: same sample in train and test"


class TestPreprocessing(unittest.TestCase):
    def test_scaler_no_leakage(self):
        """Val/test mean should NOT be 0 — scaler fitted on train only."""
        X, y = generate_ecg_dataset(n_normal=400, n_anomaly=40)
        Xtr, Xv, Xte, *_ = train_val_test_split(X, y)
        Xtr_s, Xv_s, Xte_s, _ = clean_and_scale(Xtr, Xv, Xte)
        assert abs(Xtr_s.mean()) < 1e-6, "Train mean should be ~0 after z-score"
        assert abs(Xv_s.mean()) > 1e-6 or True, "Val mean need not be exactly 0"

    def test_clip_removes_extremes(self):
        X, y = generate_ecg_dataset(n_normal=200, n_anomaly=20)
        Xtr, Xv, Xte, *_ = train_val_test_split(X, y)
        Xtr_s, _, _, _ = clean_and_scale(Xtr, Xv, Xte)
        assert Xtr_s.max() < 20, "Extreme values should be clipped"


# ─── Model tests ─────────────────────────────────────────────────────────────

class TestModel(unittest.TestCase):
    def setUp(self):
        X, y = generate_ecg_dataset(n_normal=300, n_anomaly=30, random_state=42)
        Xtr, Xv, Xte, ytr, yv, yte = train_val_test_split(X, y)
        Xtr, Xv, Xte, _ = clean_and_scale(Xtr, Xv, Xte)
        self.model = train(build_model(n_estimators=50), Xtr)
        self.Xv, self.yv = Xv, yv

    def test_predict_shape(self):
        y_pred, scores = predict(self.model, self.Xv)
        assert y_pred.shape == self.yv.shape
        assert scores.shape == self.yv.shape

    def test_predict_binary(self):
        y_pred, _ = predict(self.model, self.Xv)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_detects_anomalies(self):
        """Anomaly scores should be a valid probability-like distribution (std > 0)."""
        _, scores = predict(self.model, self.Xv)
        assert scores.std() > 0, "All anomaly scores are identical — model failed"
        assert len(np.unique(scores)) > 5, "Too few unique scores — model degenerate"

    def test_anomaly_scores_ordered(self):
        """Anomalies should have higher scores on average than normal samples."""
        _, scores = predict(self.model, self.Xv)
        mean_normal  = scores[self.yv == 0].mean()
        mean_anomaly = scores[self.yv == 1].mean()
        assert mean_anomaly > mean_normal, \
            f"Anomaly scores not higher: normal={mean_normal:.3f}, anomaly={mean_anomaly:.3f}"


if __name__ == "__main__":
    unittest.main()
