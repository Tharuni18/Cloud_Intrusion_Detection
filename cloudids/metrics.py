"""Evaluation metrics for intrusion detection (binary: attack vs benign)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

from .data import BENIGN


def binary_metrics(y_str: pd.Series, flag: np.ndarray, score: np.ndarray) -> dict:
    y = (y_str.values != BENIGN).astype(int)
    flag = np.asarray(flag).astype(int)
    tn = int(((flag == 0) & (y == 0)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    return {
        "precision": precision_score(y, flag, zero_division=0),
        "recall": recall_score(y, flag, zero_division=0),
        "f1": f1_score(y, flag, zero_division=0),
        "fpr": fp / max(fp + tn, 1),
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
    }


def per_class_recall(y_str: pd.Series, flag: np.ndarray) -> dict:
    flag = np.asarray(flag)
    out = {}
    for c in sorted(y_str.unique()):
        m = (y_str.values == c)
        out[c] = float((~flag[m]).mean()) if c == BENIGN else float(flag[m].mean())
    return out  # for benign this is the true-negative rate (1 - FPR)


def permutation_importance_pr(detector, X: pd.DataFrame, y_str: pd.Series, n: int = 3000,
                              seed: int = 0) -> pd.Series:
    """Model-agnostic importance: drop in PR-AUC when one feature is shuffled."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xs, ys = X.iloc[idx].reset_index(drop=True), y_str.iloc[idx].reset_index(drop=True)
    yb = (ys.values != BENIGN).astype(int)
    base = average_precision_score(yb, detector.decision(Xs)[1])
    drops = {}
    for c in X.columns:
        Xp = Xs.copy()
        Xp[c] = rng.permutation(Xp[c].values)
        drops[c] = base - average_precision_score(yb, detector.decision(Xp)[1])
    return pd.Series(drops).sort_values(ascending=False)
