"""Model definitions. Every model takes the same feature frame X (see build_frame)
and is a scikit-learn Pipeline, so learned steps are fitted on training rows only.

  m0_majority     predicts the most frequent training label
  m1_tfidf        TF-IDF (1-2 grams) -> logistic regression
  m2_minilm       MiniLM sentence embeddings -> logistic regression
  m3_tfidf_vader  TF-IDF (m1's best settings) + standardised VADER scores -> logistic regression

Models 1-3 share one classifier (logistic regression), so differences between
them come from the representation, not the learning algorithm.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import VADER_COLS, make_tfidf

MODEL_NAMES = ["m0_majority", "m1_tfidf", "m2_minilm", "m3_tfidf_vader"]
MODEL_LABELS = {
    "m0_majority": "M0 Majority",
    "m1_tfidf": "M1 TF-IDF",
    "m2_minilm": "M2 MiniLM",
    "m3_tfidf_vader": "M3 TF-IDF+VADER",
}


def build_frame(df: pd.DataFrame, emb: np.ndarray, vader: pd.DataFrame) -> pd.DataFrame:
    """One row per review: cleaned text, 5 VADER columns, 384 embedding columns."""
    emb_cols = [f"emb_{i}" for i in range(emb.shape[1])]
    return pd.concat([
        df[["text_clean"]].reset_index(drop=True),
        vader.reset_index(drop=True),
        pd.DataFrame(emb, columns=emb_cols),
    ], axis=1)


def _logreg(C: float, seed: int) -> LogisticRegression:
    return LogisticRegression(C=C, solver="liblinear", max_iter=5000, random_state=seed)


def build_model(name: str, params: dict, seed: int, n_emb: int = 384) -> Pipeline:
    if name == "m0_majority":
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])

    if name == "m1_tfidf":
        feats = ColumnTransformer([
            ("tfidf", make_tfidf(params["min_df"], params["sublinear_tf"]), "text_clean"),
        ])
        return Pipeline([("features", feats), ("clf", _logreg(params["C"], seed))])

    if name == "m2_minilm":
        feats = ColumnTransformer([
            ("emb", "passthrough", [f"emb_{i}" for i in range(n_emb)]),
        ])
        return Pipeline([("features", feats), ("clf", _logreg(params["C"], seed))])

    if name == "m3_tfidf_vader":
        feats = ColumnTransformer(
            [("tfidf", make_tfidf(params["min_df"], params["sublinear_tf"]), "text_clean"),
             ("vader", StandardScaler(), VADER_COLS)],
            transformer_weights={"tfidf": 1.0, "vader": params["vader_weight"]},
        )
        return Pipeline([("features", feats), ("clf", _logreg(params["C"], seed))])

    raise ValueError(f"Unknown model: {name}")


def param_grid(name: str, cfg: dict, m1_best: dict | None = None) -> list[dict]:
    """Expand the config grid into a list of parameter dicts, in a fixed order."""
    if name == "m0_majority":
        return [{}]
    grid = dict(cfg["models"][name])
    if name == "m3_tfidf_vader":
        if m1_best is None:
            raise ValueError("m3 needs m1's best TF-IDF settings")
        grid["min_df"] = [m1_best["min_df"]]
        grid["sublinear_tf"] = [m1_best["sublinear_tf"]]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]
