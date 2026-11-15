"""Stage 6: robustness checks on train + validation only (the test set is NOT used).

Usage (from the project root, after 03_tune.py):
    python scripts/07_robustness.py

1. Repeated cross-validation (5 folds x 5 repeats, stratified by deceptive x polarity)
   with each model's chosen settings. Gives 25 macro-F1 scores per model and polarity,
   showing whether the single test-set result is stable or a one-off.
2. Cross-polarity transfer (compares with Ott et al., 2013, Table 2):
   train on all positive train+val reviews, test on all negative train+val reviews, and vice versa.

Outputs
    results/tables/cv_scores.csv          every fold's macro-F1
    results/tables/cv_summary.csv         mean, SD, and how often negative > positive
    results/tables/cross_polarity.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_config  # noqa: E402
from src.evaluate import point_metrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_features, project_path, set_seed  # noqa: E402

LEARNED = ["m1_tfidf", "m2_minilm", "m3_tfidf_vader"]


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    tables = project_path(cfg["paths"]["tables"])
    best = json.loads((project_path("results") / "best_params.json").read_text())

    table, X, y, meta = load_features(cfg)
    keep = table["partition"].isin(["train", "val"]).to_numpy()
    table, X, y = table[keep].reset_index(drop=True), X[keep].reset_index(drop=True), y[keep]
    pol = table["polarity"].to_numpy()
    print(f"Using train + validation only: {len(table)} reviews (test set untouched)\n")

    # ---- 1. repeated stratified CV
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, table["cell"])):
        for name in LEARNED:
            model = build_model(name, best[name]["params"], seed, n_emb=meta["dim"]).fit(X.iloc[tr], y[tr])
            p = model.predict(X.iloc[te])
            for subset in ("all", "positive", "negative"):
                m = np.ones(len(te), bool) if subset == "all" else pol[te] == subset
                rows.append({"fold": fold, "model": name, "subset": subset,
                             "macro_f1": point_metrics(y[te][m], p[m])["macro_f1"]})
    scores = pd.DataFrame(rows)
    scores.round(4).to_csv(tables / "cv_scores.csv", index=False)

    wide = scores.pivot_table(index=["model", "fold"], columns="subset", values="macro_f1")
    wide["neg_minus_pos"] = wide["negative"] - wide["positive"]
    summary = wide.groupby("model").agg(
        all_mean=("all", "mean"), all_sd=("all", "std"),
        positive_mean=("positive", "mean"), positive_sd=("positive", "std"),
        negative_mean=("negative", "mean"), negative_sd=("negative", "std"),
        neg_minus_pos_mean=("neg_minus_pos", "mean"), neg_minus_pos_sd=("neg_minus_pos", "std"),
        folds_negative_higher=("neg_minus_pos", lambda d: int((d > 0).sum())),
    ).loc[LEARNED]
    summary["folds"] = wide.groupby("model").size().loc[LEARNED]
    summary.round(4).to_csv(tables / "cv_summary.csv")

    # ---- 2. cross-polarity transfer
    transfer = []
    for name in LEARNED:
        for train_pol, test_pol in (("positive", "negative"), ("negative", "positive")):
            tr, te = pol == train_pol, pol == test_pol
            model = build_model(name, best[name]["params"], seed, n_emb=meta["dim"]).fit(X[tr], y[tr])
            m = point_metrics(y[te], model.predict(X[te]))
            transfer.append({"model": name, "train_on": train_pol, "test_on": test_pol,
                             "n_train": int(tr.sum()), "n_test": int(te.sum()),
                             "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4)})
    transfer = pd.DataFrame(transfer)
    transfer.to_csv(tables / "cross_polarity.csv", index=False)

    show = summary[["all_mean", "positive_mean", "negative_mean", "neg_minus_pos_mean",
                    "neg_minus_pos_sd", "folds_negative_higher", "folds"]].round(3)
    print("Repeated CV (5 folds x 5 repeats), macro-F1")
    print(show.to_string(), "\n")
    print("Cross-polarity transfer (train on one polarity, test on the other)")
    print(transfer.to_string(index=False))
    


if __name__ == "__main__":
    main()
