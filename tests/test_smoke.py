import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from cloudids.data import make_synthetic, split_dataset, BENIGN
from cloudids.experiments import fit_detectors, evaluate


def test_pipeline_runs_and_beats_chance():
    ds = make_synthetic(6000, seed=1)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_dataset(ds, seed=1)
    # no row appears in more than one split
    assert len(set(map(tuple, Xtr.round(6).values)) & set(map(tuple, Xte.round(6).values))) == 0
    dets = fit_detectors(Xtr, ytr, Xva, yva, seed=1, fast=True)
    cw, pcr, _ = evaluate(dets, Xte, yte)
    assert set(cw.index) == {"Rules", "RandomForest", "XGBoost", "Autoencoder",
                             "Hybrid-OR", "Hybrid-Stacked"}
    assert cw.loc["XGBoost", "pr_auc"] > 0.8
    assert cw.loc["Autoencoder", "roc_auc"] > 0.7
    assert 0 <= cw["fpr"].max() <= 1


def test_zero_day_leaves_class_out_of_training():
    ds = make_synthetic(4000, seed=2)
    (Xtr, ytr), (Xva, yva), _ = split_dataset(ds, seed=2)
    m = (ytr != "botnet").values
    dets = fit_detectors(Xtr[m].reset_index(drop=True), ytr[m].reset_index(drop=True),
                         Xva[(yva != "botnet").values].reset_index(drop=True),
                         yva[(yva != "botnet").values].reset_index(drop=True), fast=True)
    assert "botnet" not in set(dets["XGBoost"].le_.classes_)
