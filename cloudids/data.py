"""Data loading for the cloud intrusion-detection project.

Three sources are supported:
  * ``synthetic``   - a cloud-flavoured flow generator (for development / CI / demos)
  * ``cicids2017``  - CIC-IDS2017 CSVs (MachineLearningCVE or TrafficLabelling)
  * ``unsw-nb15``   - UNSW-NB15 training/testing-set CSVs

Every loader returns a :class:`Dataset` with a numeric feature frame ``X`` and a
string label series ``y`` where benign traffic is labelled ``"benign"`` and each
attack is mapped to a coarse *family* (dos, portscan, bruteforce, ...).
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BENIGN = "benign"


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def attack_classes(self) -> list[str]:
        return sorted(c for c in self.y.unique() if c != BENIGN)

    def summary(self) -> pd.Series:
        return self.y.value_counts()


# --------------------------------------------------------------------------- #
# Synthetic cloud-flow generator
# --------------------------------------------------------------------------- #
SYN_FEATURES = [
    "duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "pkt_len_mean",
    "pkt_len_std", "iat_mean", "iat_std", "pkts_per_sec", "bytes_per_sec",
    "syn_count", "rst_count", "ack_ratio", "dst_port", "unique_dst_ports_1m",
    "conn_per_min_src", "failed_auth_count", "auth_attempts_per_min",
    "bytes_out_in_ratio", "api_calls_per_min", "new_geo_score", "off_hours",
    "tls_used",
]
_BINARY = {"off_hours", "tls_used"}
_NO_BLEND = {"dst_port"}


def _benign_primitives(m: int, rng: np.random.Generator) -> dict:
    """Benign cloud traffic: web / ssh-admin / database / API-gateway sessions."""
    mode = rng.choice(4, size=m, p=[0.50, 0.15, 0.10, 0.25])
    pick = lambda a: np.asarray(a, dtype=float)[mode]
    d = {}
    d["duration"] = rng.lognormal(pick([1.0, 3.5, 2.0, 0.3]), 0.9)
    jitter = lambda: rng.lognormal(0, 0.4, m)
    d["fwd_pkts"] = rng.poisson(pick([12, 40, 25, 6]) * jitter()) + 1.0
    d["bwd_pkts"] = rng.poisson(pick([18, 38, 30, 6]) * jitter()) + 1.0
    d["pkt_len_mean"] = np.clip(rng.normal(pick([600, 120, 400, 300]), 80), 40, 1500)
    d["fwd_mult"] = rng.uniform(0.3, 1.0, m)
    d["bwd_mult"] = rng.uniform(0.5, 1.5, m)
    d["iat_cv"] = rng.uniform(0.5, 2.0, m)
    d["syn_count"] = rng.poisson(1.2, m).astype(float)
    d["rst_count"] = rng.poisson(0.1, m).astype(float)
    d["ack_ratio"] = rng.beta(8, 2, m)
    d["dst_port"] = np.where(mode == 0, rng.choice([80, 443], m),
                             pick([0, 22, 5432, 8080])).astype(float)
    d["unique_dst_ports_1m"] = rng.poisson(pick([2, 1, 1, 2])) + 1.0
    d["conn_per_min_src"] = rng.lognormal(np.log(pick([6, 2, 4, 10])), 0.6)
    d["failed_auth_count"] = rng.poisson(pick([0.05, 0.3, 0.05, 0.05])).astype(float)
    d["auth_attempts_per_min"] = rng.poisson(pick([0.2, 1.5, 0.2, 0.2])).astype(float)
    d["api_calls_per_min"] = rng.lognormal(np.log(pick([2, 0.2, 0.5, 30])), 0.7)
    d["new_geo_score"] = rng.beta(1, 12, m)
    d["off_hours"] = (rng.random(m) < 0.2).astype(float)
    d["tls_used"] = (rng.random(m) < pick([0.9, 0.0, 0.3, 0.95])).astype(float)
    return d


def _finalize(d: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = len(d["duration"])
    d = dict(d)
    d.setdefault("fwd_bytes", d["fwd_pkts"] * d["pkt_len_mean"] * d["fwd_mult"])
    d.setdefault("bwd_bytes", d["bwd_pkts"] * d["pkt_len_mean"] * d["bwd_mult"])
    d["pkt_len_std"] = d["pkt_len_mean"] * rng.uniform(0.2, 0.9, n)
    total_pkts = d["fwd_pkts"] + d["bwd_pkts"]
    dur = np.clip(d["duration"], 1e-3, None)
    d["iat_mean"] = dur / total_pkts * 1000.0
    d["iat_std"] = d["iat_mean"] * d["iat_cv"]
    d["pkts_per_sec"] = total_pkts / dur
    d["bytes_per_sec"] = (d["fwd_bytes"] + d["bwd_bytes"]) / dur
    d["bytes_out_in_ratio"] = d["bwd_bytes"] / np.clip(d["fwd_bytes"], 1, None)
    return pd.DataFrame({k: np.asarray(d[k], dtype=float) for k in SYN_FEATURES})


def _attack(cls: str, m: int, rng: np.random.Generator) -> dict:
    d = _benign_primitives(m, rng)
    if cls == "dos":
        d["duration"] = rng.lognormal(-1.0, 0.7, m)
        d["fwd_pkts"] = rng.poisson(rng.lognormal(np.log(220), 0.5, m)) + 1.0
        d["bwd_pkts"] = rng.poisson(3, m) + 0.0
        d["pkt_len_mean"] = np.clip(rng.normal(120, 60, m), 40, 600)
        d["syn_count"] = rng.poisson(40, m).astype(float)
        d["ack_ratio"] = rng.beta(2, 6, m)
        d["conn_per_min_src"] = rng.lognormal(np.log(400), 0.7, m)
        d["dst_port"] = rng.choice([80, 443], m).astype(float)
    elif cls == "portscan":
        d["duration"] = rng.lognormal(-3.0, 0.8, m)
        d["fwd_pkts"] = rng.integers(1, 4, m).astype(float)
        d["bwd_pkts"] = rng.integers(0, 2, m).astype(float)
        d["pkt_len_mean"] = np.clip(rng.normal(60, 10, m), 40, 100)
        d["unique_dst_ports_1m"] = rng.lognormal(np.log(150), 0.8, m)
        d["rst_count"] = rng.poisson(2.0, m).astype(float)
        d["syn_count"] = rng.integers(1, 3, m).astype(float)
        d["conn_per_min_src"] = rng.lognormal(np.log(200), 0.6, m)
        d["dst_port"] = rng.integers(1, 65535, m).astype(float)
    elif cls == "bruteforce":
        d["duration"] = rng.lognormal(1.5, 0.5, m)
        d["fwd_pkts"] = rng.poisson(14, m) + 2.0
        d["bwd_pkts"] = rng.poisson(14, m) + 2.0
        d["pkt_len_mean"] = np.clip(rng.normal(110, 25, m), 40, 300)
        d["failed_auth_count"] = rng.poisson(25, m).astype(float)
        d["auth_attempts_per_min"] = rng.lognormal(np.log(40), 0.5, m)
        d["conn_per_min_src"] = rng.lognormal(np.log(30), 0.5, m)
        d["dst_port"] = rng.choice([22, 3389, 443], m, p=[0.5, 0.3, 0.2]).astype(float)
    elif cls == "exfiltration":  # data breach
        d["duration"] = rng.lognormal(4.0, 0.6, m)
        d["bwd_pkts"] = rng.poisson(4000, m) + 1.0
        d["pkt_len_mean"] = np.clip(rng.normal(1200, 150, m), 400, 1500)
        d["bwd_bytes"] = rng.lognormal(np.log(3e7), 1.0, m)
        d["off_hours"] = (rng.random(m) < 0.75).astype(float)
        d["new_geo_score"] = rng.beta(5, 3, m)
        d["tls_used"] = (rng.random(m) < 0.95).astype(float)
        d["api_calls_per_min"] = rng.lognormal(np.log(15), 0.7, m)
    elif cls == "botnet":  # C2 beaconing
        d["duration"] = rng.lognormal(3.0, 0.3, m)
        d["fwd_pkts"] = rng.poisson(5, m) + 1.0
        d["bwd_pkts"] = rng.poisson(5, m) + 1.0
        d["pkt_len_mean"] = np.clip(rng.normal(150, 30, m), 40, 400)
        d["iat_cv"] = rng.uniform(0.02, 0.1, m)  # very regular timing
        d["conn_per_min_src"] = rng.lognormal(np.log(3), 0.2, m)
        d["dst_port"] = rng.choice([4444, 6667, 8443, 1337], m).astype(float)
        d["new_geo_score"] = rng.beta(3, 4, m)
        d["off_hours"] = (rng.random(m) < 0.5).astype(float)
    else:
        raise ValueError(cls)
    return d


def make_synthetic(n: int = 40000, seed: int = 42, stealth: float = 0.12) -> Dataset:
    """Generate a labelled cloud-flow dataset.

    ``stealth`` is the fraction of attack flows that are blended back toward benign
    behaviour (in log space) so classes overlap and results are not trivially perfect.
    """
    rng = np.random.default_rng(seed)
    props = {BENIGN: 0.70, "dos": 0.08, "portscan": 0.06, "bruteforce": 0.06,
             "exfiltration": 0.05, "botnet": 0.05}
    frames, labels = [], []
    for cls, p in props.items():
        m = int(n * p)
        if cls == BENIGN:
            df = _finalize(_benign_primitives(m, rng), rng)
        else:
            df = _finalize(_attack(cls, m, rng), rng)
            k = int(m * stealth)
            if k:
                ben = _finalize(_benign_primitives(k, rng), rng)
                alpha = rng.uniform(0.15, 0.55, size=(k, 1))
                cols = [c for c in SYN_FEATURES if c not in _BINARY | _NO_BLEND]
                a = df.iloc[:k][cols].to_numpy()
                b = ben[cols].to_numpy()
                blended = np.expm1(alpha * np.log1p(a) + (1 - alpha) * np.log1p(b))
                df.iloc[:k, [df.columns.get_loc(c) for c in cols]] = blended
        frames.append(df)
        labels += [cls] * len(df)
    X = pd.concat(frames, ignore_index=True)
    y = pd.Series(labels, name="label")
    perm = rng.permutation(len(X))
    return Dataset("synthetic", X.iloc[perm].reset_index(drop=True),
                   y.iloc[perm].reset_index(drop=True))


# --------------------------------------------------------------------------- #
# CIC-IDS2017
# --------------------------------------------------------------------------- #
_CIC_DROP = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp",
             "Fwd Header Length.1", "SimillarHTTP"]


def _cic_family(label: str) -> str:
    l = str(label).strip().lower()
    if l == "benign":
        return BENIGN
    if "ddos" in l or l.startswith("dos"):
        return "dos"
    if "portscan" in l:
        return "portscan"
    if "patator" in l:
        return "bruteforce"
    if "web attack" in l:
        return "webattack"
    if l == "bot":
        return "botnet"
    if "infiltration" in l:
        return "infiltration"
    if "heartbleed" in l:
        return "heartbleed"
    return l.replace(" ", "_")


def _cap(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if not cap:
        return df
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), cap), random_state=seed)))


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in df.columns if c != "label"]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def load_cicids2017(path: str, cap_per_class: int | None = 40000, seed: int = 42) -> Dataset:
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(f"No CSV files found at {path}")
    parts = []
    for f in files:
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.drop(columns=[c for c in _CIC_DROP if c in df.columns])
        df["label"] = df["Label"].map(_cic_family)
        df = df.drop(columns=["Label"])
        df = _clean_numeric(df).drop_duplicates()
        parts.append(_cap(df, cap_per_class, seed))      # per-file cap keeps memory bounded
        print(f"  loaded {os.path.basename(f)}: {len(df):,} rows (after cleaning)")
    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    df = _cap(df, cap_per_class, seed).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = df.pop("label")
    X = df.astype(np.float32)
    X = X.loc[:, X.nunique() > 1]
    return Dataset("cicids2017", X, y)


# --------------------------------------------------------------------------- #
# UNSW-NB15
# --------------------------------------------------------------------------- #
def load_unsw_nb15(path: str, cap_per_class: int | None = 40000, seed: int = 42,
                   top_k_categories: int = 10) -> Dataset:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*training-set*.csv"))
                       + glob.glob(os.path.join(path, "*testing-set*.csv")))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError("Expected UNSW_NB15_training-set.csv / testing-set.csv")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["label"] = (df["attack_cat"].astype(str).str.strip().str.lower()
                   .replace({"normal": BENIGN, "nan": BENIGN, "": BENIGN}))
    df = df.drop(columns=[c for c in ["id", "attack_cat"] if c in df.columns])
    # the binary 'label' column of the original file was overwritten above; drop its twin
    cat_cols = [c for c in ["proto", "service", "state"] if c in df.columns]
    for c in cat_cols:
        top = df[c].value_counts().nlargest(top_k_categories).index
        df[c] = df[c].where(df[c].isin(top), "other")
    df = pd.get_dummies(df, columns=cat_cols, dtype=np.float32)
    df = _clean_numeric(df).drop_duplicates()
    df = _cap(df, cap_per_class, seed).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = df.pop("label")
    X = df.astype(np.float32)
    X = X.loc[:, X.nunique() > 1]
    return Dataset("unsw-nb15", X, y)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def load_dataset(name: str, path: str | None = None, n_samples: int = 40000,
                 cap_per_class: int | None = 40000, seed: int = 42) -> Dataset:
    if name == "synthetic":
        return make_synthetic(n_samples, seed)
    if path is None:
        raise ValueError(f"--data-path is required for dataset '{name}'")
    if name == "cicids2017":
        return load_cicids2017(path, cap_per_class, seed)
    if name in ("unsw-nb15", "unsw"):
        return load_unsw_nb15(path, cap_per_class, seed)
    raise ValueError(f"Unknown dataset {name}")


def split_dataset(ds: Dataset, seed: int = 42, val_size: float = 0.2, test_size: float = 0.2,
                  min_class_count: int = 10):
    """Stratified train/val/test split (default 60/20/20) on *de-duplicated* rows.

    Returns ``(Xtr, ytr), (Xva, yva), (Xte, yte)``. Classes with fewer than
    ``min_class_count`` rows are dropped (they cannot be split meaningfully).
    """
    X, y = ds.X.copy(), ds.y.copy()
    key = X.assign(__y=y.values)
    keep = ~key.duplicated().to_numpy()
    X, y = X[keep], y[keep]
    counts = y.value_counts()
    small = counts[counts < min_class_count].index
    if len(small):
        warnings.warn(f"Dropping classes with <{min_class_count} samples: {list(small)}")
        m = ~y.isin(small)
        X, y = X[m], y[m]
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=test_size, stratify=y,
                                            random_state=seed)
    rel_val = val_size / (1 - test_size)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=rel_val, stratify=ytmp,
                                          random_state=seed)
    rs = lambda a: a.reset_index(drop=True)
    return (rs(Xtr), rs(ytr)), (rs(Xva), rs(yva)), (rs(Xte), rs(yte))
