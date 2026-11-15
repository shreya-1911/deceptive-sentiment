"""Stage 2: choose hyperparameters for every model on the VALIDATION set.

Usage (from the project root):
    python scripts/03_tune.py

Each candidate is fitted on the training partition and scored on validation
(macro-F1). The test partition is not touched here.

Outputs
    results/best_params.json              chosen settings per model
    results/tables/tuning_results.csv     every candidate's validation scores
    results/tables/minilm_truncation.csv  reviews longer than MiniLM's input limit, per cell
    results/tables/vader_by_cell.csv      mean VADER scores per cell (training data only)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_config  # noqa: E402
from src.evaluate import point_metrics  # noqa: E402
from src.features import VADER_COLS  # noqa: E402
from src.models import MODEL_NAMES, build_model, param_grid  # noqa: E402
from src.utils import load_features, project_path, set_seed  # noqa: E402


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    tables = project_path(cfg["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)

    table, X, y, meta = load_features(cfg)
    tr = (table["partition"] == "train").to_numpy()
    va = (table["partition"] == "val").to_numpy()
    print(f"Train: {tr.sum()}   Validation: {va.sum()}   (test untouched)\n")

    # --- descriptive tables for the report
    trunc = pd.DataFrame(meta["truncated_by_cell"]).T.rename(columns={"sum": "truncated", "count": "reviews"})
    trunc["share"] = (trunc["truncated"] / trunc["reviews"]).round(3)
    trunc["mean_tokens"] = pd.Series(meta["mean_tokens_by_cell"])
    trunc.to_csv(tables / "minilm_truncation.csv")
    print(f"MiniLM input limit: {meta['max_seq_length']} tokens. Reviews over the limit:")
    print(trunc.to_string(), "\n")

    vader_tr = X.loc[tr, VADER_COLS].assign(cell=table.loc[tr, "cell"].to_numpy())
    vader_by_cell = vader_tr.groupby("cell").mean().round(3)
    vader_by_cell.to_csv(tables / "vader_by_cell.csv")
    print("Mean VADER scores by cell (training data):")
    print(vader_by_cell.to_string(), "\n")

    # --- grid search on validation
    rows, best = [], {}
    for name in MODEL_NAMES:
        grid = param_grid(name, cfg, best.get("m1_tfidf", {}).get("params"))
        scored = []
        for params in grid:
            model = build_model(name, params, seed, n_emb=meta["dim"])
            model.fit(X[tr], y[tr])
            m = point_metrics(y[va], model.predict(X[va]))
            scored.append((m["macro_f1"], params, m))
            rows.append({"model": name, "params": json.dumps(params), **{f"val_{k}": v for k, v in m.items()}})
        top = max(scored, key=lambda s: s[0])  # max() keeps the first of tied scores
        best[name] = {"params": top[1], "val_macro_f1": round(top[0], 4)}
        print(f"{name:16s} {len(grid):2d} candidates  best val macro-F1 {top[0]:.3f}  {top[1]}")

    pd.DataFrame(rows).to_csv(tables / "tuning_results.csv", index=False)
    (project_path("results") / "best_params.json").write_text(json.dumps(best, indent=2))
    


if __name__ == "__main__":
    main()
