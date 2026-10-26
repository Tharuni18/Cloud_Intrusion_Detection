"""Detectors used in the project.

Every detector exposes ``decision(X_df) -> (flag: bool[n], score: float[n])`` where
``flag`` is the alert at the detector's operating point and ``score`` is a continuous
"attack-ness" value used for PR/ROC curves.

Backends degrade gracefully:
  * XGBoost  -> sklearn HistGradientBoosting if ``xgboost`` is not installed
  * PyTorch  -> sklearn MLPRegressor autoencoder if ``torch`` is not installed
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .data import BENIGN

try:  # optional dependencies
    import xgboost as _xgb  # type: ignore
except Exception:  # pragma: no cover
    _xgb = None
try:
    import torch as _torch  # type: ignore
except Exception:  # pragma: no cover
    _torch = None


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class Preprocessor:
    """Signed log1p (tames heavy-tailed network features) + z-score + clipping.

    Fit on *benign training traffic only*, so no attack statistics leak into the
    scaler that the unsupervised model relies on.
    """

    CLIP = 20.0

    @staticmethod
    def _slog(a: np.ndarray) -> np.ndarray:
        return np.sign(a) * np.log1p(np.abs(a))

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.columns = list(X.columns)
        Z = self._slog(X.to_numpy(dtype=np.float64))
        self.mean_ = Z.mean(axis=0)
        self.std_ = np.maximum(Z.std(axis=0), 1e-2)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._slog(X[self.columns].to_numpy(dtype=np.float64))
        return np.clip((Z - self.mean_) / self.std_, -self.CLIP, self.CLIP).astype(np.float32)


# --------------------------------------------------------------------------- #
# Baseline: static, signature-style rules
# --------------------------------------------------------------------------- #
_RULE_KEYWORDS = ["packets/s", "bytes/s", "pkts_per_sec", "bytes_per_sec", "syn", "rst",
                  "failed_auth", "auth_attempts", "unique_dst_ports", "conn_per_min",
                  "rate", "sload", "dload", "api_calls"]


class RuleDetector:
    """Fixed-threshold baseline that mimics a hand-written rule set.

    For a handful of rate/volume/flag features (matched by column-name keywords) the
    threshold is frozen at a high quantile of benign training data; a flow is flagged
    if *any* rule fires. It never adapts afterwards, which is the point of the baseline.
    """

    def __init__(self, quantile: float = 0.995):
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RuleDetector":
        ben = X[y.values == BENIGN]
        cols = [c for c in X.columns if any(k in c.lower() for k in _RULE_KEYWORDS)]
        self.thresholds_ = {c: float(ben[c].quantile(self.quantile)) for c in cols}
        return self

    def decision(self, X: pd.DataFrame):
        fired = np.zeros(len(X))
        for c, t in self.thresholds_.items():
            fired += (X[c].to_numpy() > t)
        return fired > 0, fired


# --------------------------------------------------------------------------- #
# Supervised detectors
# --------------------------------------------------------------------------- #
def _make_classifier(kind: str, n_classes: int, seed: int, fast: bool = False):
    if kind == "rf":
        return (RandomForestClassifier(n_estimators=100 if fast else 200, max_depth=None,
                                       min_samples_leaf=2, n_jobs=-1, random_state=seed,
                                       class_weight="balanced_subsample"), "scikit-learn RF")
    if kind == "xgb":
        if _xgb is not None:
            m = _xgb.XGBClassifier(n_estimators=150 if fast else 300, max_depth=6,
                                   learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                   tree_method="hist", objective="multi:softprob",
                                   num_class=n_classes, n_jobs=-1, random_state=seed,
                                   eval_metric="mlogloss")
            return m, "xgboost"
        m = HistGradientBoostingClassifier(max_iter=100 if fast else 200, learning_rate=0.1,
                                           max_depth=8, random_state=seed)
        return m, "sklearn HistGradientBoosting (XGBoost fallback)"
    raise ValueError(kind)


def _fit_classifier(model, Z, yi, kind):
    # sqrt-balanced weights: help rare classes without letting a 10-sample class dominate
    w = compute_sample_weight("balanced", yi) ** 0.5
    if kind == "rf":
        model.fit(Z, yi)
    else:
        model.fit(Z, yi, sample_weight=w)


class SupervisedDetector:
    """Multi-class classifier; attack score = 1 - P(benign)."""

    def __init__(self, kind: str, prep: Preprocessor, seed: int = 42, threshold: float = 0.5,
                 fast: bool = False):
        self.kind, self.prep, self.seed, self.threshold, self.fast = kind, prep, seed, threshold, fast

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupervisedDetector":
        self.le_ = LabelEncoder().fit(y)
        yi = self.le_.transform(y)
        self.benign_idx_ = int(np.where(self.le_.classes_ == BENIGN)[0][0])
        self.model_, self.backend = _make_classifier(self.kind, len(self.le_.classes_),
                                                     self.seed, self.fast)
        _fit_classifier(self.model_, self.prep.transform(X), yi, self.kind)
        return self

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.prep.transform(X))

    def attack_score(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 - self.proba(X)[:, self.benign_idx_]

    def predict_label(self, X: pd.DataFrame) -> np.ndarray:
        return self.le_.classes_[self.proba(X).argmax(1)]

    def decision(self, X: pd.DataFrame):
        s = self.attack_score(X)
        return s >= self.threshold, s


# --------------------------------------------------------------------------- #
# Autoencoder (unsupervised anomaly detector)
# --------------------------------------------------------------------------- #
class AEDetector:
    """Autoencoder trained on benign traffic only.

    Anomaly score = per-flow reconstruction MSE. The alert threshold is the
    ``percentile``-th percentile of benign *validation* errors, i.e. it is calibrated
    for a target false-positive rate (default ~1%) without ever seeing an attack.
    """

    def __init__(self, prep: Preprocessor, seed: int = 42, epochs: int = 60,
                 percentile: float = 99.0, backend: str = "auto", fast: bool = False):
        self.prep, self.seed, self.percentile = prep, seed, percentile
        self.epochs = 25 if fast else epochs
        self.backend = ("torch" if _torch is not None else "sklearn") if backend == "auto" else backend

    # -- training ---------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AEDetector":
        Z = self.prep.transform(X[y.values == BENIGN])
        d = Z.shape[1]
        h1, h2, b = max(32, d), max(16, d // 2), max(4, d // 4)
        if self.backend == "torch":
            self._fit_torch(Z, d, h1, h2, b)
        else:
            self.model_ = MLPRegressor(hidden_layer_sizes=(h1, h2, b, h2, h1), activation="relu",
                                       solver="adam", early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=8, max_iter=self.epochs,
                                       random_state=self.seed)
            self.model_.fit(Z, Z)
        return self

    def _fit_torch(self, Z, d, h1, h2, b):  # pragma: no cover (needs torch)
        nn = _torch.nn
        _torch.manual_seed(self.seed)
        net = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(),
                            nn.Linear(h2, b), nn.ReLU(), nn.Linear(b, h2), nn.ReLU(),
                            nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, d))
        Ztr, Zva = train_test_split(Z, test_size=0.1, random_state=self.seed)
        Ztr, Zva = _torch.from_numpy(Ztr), _torch.from_numpy(Zva)
        opt = _torch.optim.Adam(net.parameters(), lr=1e-3)
        best, best_state, patience = np.inf, None, 0
        for _ in range(self.epochs):
            net.train()
            perm = _torch.randperm(len(Ztr))
            for i in range(0, len(Ztr), 256):
                xb = Ztr[perm[i:i + 256]]
                opt.zero_grad()
                loss = ((net(xb) - xb) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with _torch.no_grad():
                v = float(((net(Zva) - Zva) ** 2).mean())
            if v < best - 1e-6:
                best, patience = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.model_ = net

    # -- scoring ----------------------------------------------------------- #
    def error(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.prep.transform(X)
        if self.backend == "torch":  # pragma: no cover
            with _torch.no_grad():
                t = _torch.from_numpy(Z)
                return ((self.model_(t) - t) ** 2).mean(1).numpy()
        return ((self.model_.predict(Z) - Z) ** 2).mean(1)

    def calibrate(self, X: pd.DataFrame, y: pd.Series) -> "AEDetector":
        errs = self.error(X[y.values == BENIGN])
        self.val_errors_ = np.sort(errs)
        self.threshold_ = float(np.percentile(errs, self.percentile))
        return self

    def cdf(self, err: np.ndarray) -> np.ndarray:
        """Empirical CDF of benign validation error -> comparable-to-probability score."""
        return np.searchsorted(self.val_errors_, err) / len(self.val_errors_)

    def decision(self, X: pd.DataFrame):
        e = self.error(X)
        return e > self.threshold_, e


# --------------------------------------------------------------------------- #
# Hybrids
# --------------------------------------------------------------------------- #
class HybridOR:
    """Parallel ensemble: alert if the supervised model OR the autoencoder fires.

    The alert is tagged ``known`` (classifier fired) or ``novel`` (only the autoencoder
    fired), which gives analysts a useful triage signal.
    """

    def __init__(self, sup: SupervisedDetector, ae: AEDetector):
        self.sup, self.ae = sup, ae

    def decision(self, X: pd.DataFrame):
        fs, ss = self.sup.decision(X)
        fa, ea = self.ae.decision(X)
        return fs | fa, np.maximum(ss, self.ae.cdf(ea))

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        fs, _ = self.sup.decision(X)
        fa, _ = self.ae.decision(X)
        return np.where(fs, "known-attack", np.where(fa, "novel-anomaly", "benign"))


class HybridStacked:
    """Feature-level fusion: the autoencoder's reconstruction error is appended as an
    extra feature for the supervised classifier.

    The training set is split in two so the classifier never sees *in-sample* AE errors
    (which would be optimistically low): AE is fit on benign rows of half A, the
    classifier on half B.
    """

    def __init__(self, kind: str, prep: Preprocessor, seed: int = 42, fast: bool = False,
                 ae_kwargs: dict | None = None):
        self.kind, self.prep, self.seed, self.fast = kind, prep, seed, fast
        self.ae_kwargs = ae_kwargs or {}

    def _augment(self, X: pd.DataFrame) -> np.ndarray:
        err = np.log10(self.ae_.error(X) + 1e-6)[:, None].astype(np.float32)
        return np.hstack([self.prep.transform(X), err])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridStacked":
        try:
            XA, XB, yA, yB = train_test_split(X, y, test_size=0.5, stratify=y, random_state=self.seed)
        except ValueError:
            XA, XB, yA, yB = train_test_split(X, y, test_size=0.5, random_state=self.seed)
        self.ae_ = AEDetector(self.prep, seed=self.seed, fast=self.fast, **self.ae_kwargs).fit(XA, yA)
        self.le_ = LabelEncoder().fit(yB)
        yi = self.le_.transform(yB)
        self.benign_idx_ = int(np.where(self.le_.classes_ == BENIGN)[0][0])
        self.model_, self.backend = _make_classifier(self.kind, len(self.le_.classes_),
                                                     self.seed, self.fast)
        _fit_classifier(self.model_, self._augment(XB), yi, self.kind)
        return self

    def decision(self, X: pd.DataFrame):
        p = self.model_.predict_proba(self._augment(X))
        s = 1.0 - p[:, self.benign_idx_]
        return s >= 0.5, s
