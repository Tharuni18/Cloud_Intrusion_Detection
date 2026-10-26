"""Experiment orchestration: closed-world, zero-day (held-out class), latency, report."""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import plots
from .data import BENIGN, Dataset, split_dataset
from .metrics import binary_metrics, per_class_recall, permutation_importance_pr
from .models import (AEDetector, HybridOR, HybridStacked, Preprocessor, RuleDetector,
                     SupervisedDetector)


def fit_detectors(Xtr, ytr, Xva, yva, seed=42, ae_percentile=99.0, fast=False) -> OrderedDict:
    """Fit every detector on (train) and calibrate thresholds on (val)."""
    prep = Preprocessor().fit(Xtr[ytr.values == BENIGN])
    rules = RuleDetector().fit(Xtr, ytr)
    rf = SupervisedDetector("rf", prep, seed, fast=fast).fit(Xtr, ytr)
    xgb = SupervisedDetector("xgb", prep, seed, fast=fast).fit(Xtr, ytr)
    ae = AEDetector(prep, seed, percentile=ae_percentile, fast=fast).fit(Xtr, ytr).calibrate(Xva, yva)
    stacked = HybridStacked("xgb", prep, seed, fast=fast,
                            ae_kwargs={"percentile": ae_percentile}).fit(Xtr, ytr)
    return OrderedDict([
        ("Rules", rules), ("RandomForest", rf), ("XGBoost", xgb), ("Autoencoder", ae),
        ("Hybrid-OR", HybridOR(xgb, ae)), ("Hybrid-Stacked", stacked),
    ])


def evaluate(detectors, Xte, yte):
    rows, per_class, scores = [], {}, {}
    for name, det in detectors.items():
        flag, score = det.decision(Xte)
        m = binary_metrics(yte, flag, score)
        m["detector"] = name
        if isinstance(det, SupervisedDetector):
            from sklearn.metrics import f1_score
            m["macro_f1_multiclass"] = f1_score(yte, det.predict_label(Xte), average="macro")
        rows.append(m)
        per_class[name] = per_class_recall(yte, flag)
        scores[name] = score
    df = pd.DataFrame(rows).set_index("detector")
    return df, pd.DataFrame(per_class), scores


def zero_day(Xtr, ytr, Xva, yva, Xte, yte, holdouts, seed, ae_percentile, fast):
    rows = []
    for h in holdouts:
        print(f"  [zero-day] holding out '{h}' ...", flush=True)
        mt, mv = (ytr != h).values, (yva != h).values
        dets = fit_detectors(Xtr[mt].reset_index(drop=True), ytr[mt].reset_index(drop=True),
                             Xva[mv].reset_index(drop=True), yva[mv].reset_index(drop=True),
                             seed, ae_percentile, fast)
        is_h = (yte == h).values
        is_ben = (yte == BENIGN).values
        is_known = ~is_h & ~is_ben
        for name, det in dets.items():
            flag, score = det.decision(Xte)
            rows.append({"holdout": h, "detector": name,
                         "recall_holdout": float(flag[is_h].mean()),
                         "recall_known": float(flag[is_known].mean()),
                         "fpr": float(flag[is_ben].mean())})
    return pd.DataFrame(rows)


def latency(detectors, Xte, n_batch=2000, n_single=150):
    rows = []
    Xb = Xte.iloc[:n_batch]
    for name, det in detectors.items():
        t = time.perf_counter(); det.decision(Xb); tb = time.perf_counter() - t
        t = time.perf_counter()
        for i in range(n_single):
            det.decision(Xte.iloc[i:i + 1])
        ts = time.perf_counter() - t
        rows.append({"detector": name, "batch_flows_per_sec": len(Xb) / tb,
                     "single_flow_ms": 1000 * ts / n_single})
    return pd.DataFrame(rows).set_index("detector")


def _md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    if df.index.name or not isinstance(df.index, pd.RangeIndex):
        df = df.rename_axis(df.index.name or "item").reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                             for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def run_all(ds: Dataset, out_dir: str, seed=42, ae_percentile=99.0, holdouts="all",
            fast=False, skip_zero_day=False, save_model=True):
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_dataset(ds, seed)
    print(f"Dataset {ds.name}: train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,} "
          f"features={Xtr.shape[1]}")
    print(pd.concat([ytr.value_counts().rename("train"), yte.value_counts().rename("test")],
                    axis=1).to_string())

    # ---------------- 1. closed-world ------------------------------------- #
    print("\n[1/3] Closed-world evaluation ...", flush=True)
    dets = fit_detectors(Xtr, ytr, Xva, yva, seed, ae_percentile, fast)
    cw, pcr, scores = evaluate(dets, Xte, yte)
    cw.to_csv(os.path.join(out_dir, "closed_world.csv"))
    pcr.to_csv(os.path.join(out_dir, "per_class_recall.csv"))
    print(cw.round(4).to_string())

    plots.pr_curves(yte, scores, fig_dir)
    xgb = dets["XGBoost"]
    plots.confusion(yte, xgb.predict_label(Xte), list(xgb.le_.classes_), fig_dir)
    imp = permutation_importance_pr(xgb, Xte, yte)
    imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["pr_auc_drop"])
    plots.importance(imp, fig_dir)
    ae = dets["Autoencoder"]
    errs = {c: ae.error(Xte[yte.values == c]) for c in sorted(yte.unique())}
    plots.ae_error_hist(errs, ae.threshold_, fig_dir)

    # ---------------- 2. zero-day ----------------------------------------- #
    zd = None
    if not skip_zero_day:
        print("\n[2/3] Zero-day (held-out attack class) evaluation ...", flush=True)
        hs = ds.attack_classes if holdouts == "all" else [h.strip() for h in holdouts.split(",")]
        hs = [h for h in hs if h in set(ytr)]
        zd = zero_day(Xtr, ytr, Xva, yva, Xte, yte, hs, seed, ae_percentile, fast)
        zd.to_csv(os.path.join(out_dir, "zero_day.csv"), index=False)
        plots.zero_day_bars(zd, fig_dir)
        print(zd.pivot(index="holdout", columns="detector", values="recall_holdout").round(3).to_string())

    # ---------------- 3. latency ------------------------------------------ #
    print("\n[3/3] Inference latency ...", flush=True)
    lat = latency(dets, Xte)
    lat.to_csv(os.path.join(out_dir, "latency.csv"))
    print(lat.round(2).to_string())

    # ---------------- artifacts ------------------------------------------- #
    meta = {"dataset": ds.name, "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
            "n_features": int(Xtr.shape[1]), "seed": seed, "ae_percentile": ae_percentile,
            "supervised_backend_xgb": xgb.backend, "autoencoder_backend": ae.backend,
            "ae_threshold": ae.threshold_}
    json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=2)
    if save_model:
        import joblib
        joblib.dump({"detector": dets["Hybrid-OR"], "columns": list(Xtr.columns)},
                    os.path.join(out_dir, "model.joblib"))
    write_report(out_dir, meta, cw, pcr, zd, lat, imp, ds)
    print(f"\nDone. Results in {out_dir}/ (see report.md)")
    return dets, cw, zd


def write_report(out_dir, meta, cw, pcr, zd, lat, imp, ds):
    L = [f"# Cloud Intrusion Detection - Experiment Report\n",
         f"*Dataset:* `{meta['dataset']}`  |  train/val/test = {meta['n_train']:,} / "
         f"{meta['n_val']:,} / {meta['n_test']:,}  |  features = {meta['n_features']}\n",
         f"*Backends:* supervised boosting = **{meta['supervised_backend_xgb']}**, "
         f"autoencoder = **{meta['autoencoder_backend']}**; AE threshold = benign-val "
         f"p{meta['ae_percentile']:g}.\n",
         "## 1. Closed-world results (all attack types seen in training)\n",
         _md_table(cw), "",
         "![PR](figures/pr_curves.png)\n",
         "Per-class detection rate (benign column = true-negative rate):\n",
         _md_table(pcr.round(4)), ""]
    if zd is not None:
        piv = zd.pivot(index="holdout", columns="detector", values="recall_holdout")
        fpr = zd.groupby("detector")["fpr"].mean().to_frame("mean_fpr")
        L += ["## 2. Zero-day results (attack class removed from training)\n",
              "Recall on the **unseen** class (each row = a separate retraining):\n",
              _md_table(piv.round(4)), "",
              "![zero-day](figures/zero_day_recall.png)\n",
              "Mean false-positive rate across those runs:\n", _md_table(fpr), "",
              "Recall on the *known* attack classes in the same runs:\n",
              _md_table(zd.pivot(index="holdout", columns="detector", values="recall_known").round(4)), ""]
    L += ["## 3. Latency\n", _md_table(lat.round(3)), "",
          "## 4. Explainability\n", "Top features by permutation importance:\n",
          _md_table(imp.head(10).to_frame("pr_auc_drop").round(4)), "",
          "![imp](figures/feature_importance.png)\n",
          "![ae](figures/ae_error_distribution.png)\n"]
    if meta["dataset"] == "synthetic":
        L += ["> **Note:** these numbers come from the *synthetic* generator, which exists to "
              "validate the pipeline. Re-run on CIC-IDS2017 / UNSW-NB15 for results you can "
              "report.\n"]
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(L))
