"""
splitting.py — 5-fold stratified cross-validation.

Each fold reserves one fifth of the data as the test set, splits the
remaining four fifths 80/20 into train/val. All splits are stratified
on the label to preserve the class ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    for trainval_idx, test_idx in skf.split(np.zeros(len(y)), y):
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.20,
            random_state=random_state,
            stratify=y[trainval_idx],
        )
        splits.append(
            (np.array(train_idx), np.array(val_idx), np.array(test_idx))
        )
    return splits
