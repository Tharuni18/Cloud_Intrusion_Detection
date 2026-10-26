"""Figures for the report (matplotlib only)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from .data import BENIGN


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def pr_curves(y_str, scores: dict, out_dir, name="pr_curves.png"):
    y = (y_str.values != BENIGN).astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for k, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=k, lw=1.8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall (binary: attack vs benign)")
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=.3)
    return _save(fig, out_dir, name)


def zero_day_bars(df: pd.DataFrame, out_dir, name="zero_day_recall.png"):
    """df columns: holdout, detector, recall_holdout."""
    piv = df.pivot(index="holdout", columns="detector", values="recall_holdout")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    piv.plot.bar(ax=ax, width=0.82)
    ax.set_ylabel("Recall on the unseen attack class"); ax.set_xlabel("Attack class held out of training")
    ax.set_ylim(0, 1.05); ax.set_title("Zero-day test: detection of attack classes never seen in training")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    plt.setp(ax.get_xticklabels(), rotation=0); ax.grid(axis="y", alpha=.3)
    return _save(fig, out_dir, name)


def confusion(y_true, y_pred, labels, out_dir, name="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax,
                                            normalize="true", xticks_rotation=45,
                                            values_format=".2f", colorbar=False)
    ax.set_title("Supervised model: normalised confusion matrix")
    return _save(fig, out_dir, name)


def importance(imp: pd.Series, out_dir, top=15, name="feature_importance.png"):
    top_s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_s.index, top_s.values, color="#3b6ea5")
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled")
    ax.set_title("Permutation feature importance (supervised model)")
    return _save(fig, out_dir, name)


def ae_error_hist(errors_by_class: dict, threshold, out_dir, name="ae_error_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    allv = np.concatenate(list(errors_by_class.values()))
    bins = np.logspace(np.log10(max(allv.min(), 1e-6)), np.log10(allv.max()), 60)
    for k, v in errors_by_class.items():
        ax.hist(v, bins=bins, alpha=0.5 if k != BENIGN else 0.8, label=k, density=True)
    ax.axvline(threshold, color="k", ls="--", label="alert threshold")
    ax.set_xscale("log"); ax.set_xlabel("Reconstruction error (MSE, log scale)")
    ax.set_ylabel("Density"); ax.set_title("Autoencoder reconstruction error by class")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, name)
