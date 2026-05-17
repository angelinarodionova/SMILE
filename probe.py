"""
probe.py — Hallucination probe classifier.

Logistic-regression probe with StandardScaler preprocessing. Inherits
nn.Module to keep the contract required by the evaluation harness;
internally it delegates to sklearn's LogisticRegression — the standard
probe choice in the hallucination-detection literature, and a more
appropriate fit than an MLP given the small dataset size.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._clf: LogisticRegression | None = None
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Probe uses scikit-learn; call predict/predict_proba instead."
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)
        self._clf = LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
        self._clf.fit(X_scaled, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 101)])
        )
        best_threshold, best_f1 = 0.5, -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)
        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        X_scaled = self._scaler.transform(X)
        return self._clf.predict_proba(X_scaled)
