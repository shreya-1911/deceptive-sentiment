"""Metrics, bootstrap confidence intervals and McNemar's test.

Label coding: 1 = deceptive (the positive class for precision/recall), 0 = truthful.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

METRICS = {
    "accuracy": lambda y, p: accuracy_score(y, p),
    "precision": lambda y, p: precision_score(y, p, pos_label=1, zero_division=0),
    "recall": lambda y, p: recall_score(y, p, pos_label=1, zero_division=0),
    "macro_f1": lambda y, p: f1_score(y, p, average="macro", zero_division=0),
}


def point_metrics(y, p) -> dict:
    return {k: float(f(y, p)) for k, f in METRICS.items()}


def bootstrap_ci(y, p, n: int = 1000, level: float = 0.95, seed: int = 42) -> dict:
    """Percentile bootstrap CI for every metric: resample test reviews with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for k, f in METRICS.items():
        vals = np.array([f(y[i], p[i]) for i in idx])
        out[f"{k}_lo"], out[f"{k}_hi"] = (float(np.quantile(vals, lo_q)),
                                          float(np.quantile(vals, hi_q)))
    return out


def bootstrap_diff(y_a, p_a, y_b, p_b, metric: str = "macro_f1",
                   n: int = 1000, level: float = 0.95, seed: int = 42) -> dict:
    """CI for metric(group A) - metric(group B), resampling each group independently.

    Used for H2: A = positive-review test subset, B = negative-review test subset.
    """
    f = METRICS[metric]
    y_a, p_a, y_b, p_b = map(np.asarray, (y_a, p_a, y_b, p_b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        ia = rng.integers(0, len(y_a), len(y_a))
        ib = rng.integers(0, len(y_b), len(y_b))
        diffs[i] = f(y_a[ia], p_a[ia]) - f(y_b[ib], p_b[ib])
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    return {
        "diff": float(f(y_a, p_a) - f(y_b, p_b)),
        "diff_lo": float(np.quantile(diffs, lo_q)),
        "diff_hi": float(np.quantile(diffs, hi_q)),
    }


def mcnemar(y, p_a, p_b) -> dict:
    """Exact McNemar test on paired predictions from two models on the same reviews.

    b = A right, B wrong;  c = A wrong, B right.  Only disagreements carry information.
    """
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    a_ok, b_ok = p_a == y, p_b == y
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    pval = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": float(pval)}
