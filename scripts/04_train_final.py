"""Stage 3: refit each model with its chosen settings on train + validation, and save it.

Usage (from the project root):
    python scripts/04_train_final.py

Outputs
    models/<model>.joblib
    results/tables/train_times.csv   fit time on train+val (seconds)

Training time for M2 excludes computing the embeddings; that one-off cost is
recorded separately (embed_ms_per_review) from data/cache/*_meta.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_config  # noqa: E402
from src.models import MODEL_NAMES, build_model  # noqa: E402
from src.utils import load_features, project_path, set_seed  # noqa: E402


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    best = json.loads((project_path("results") / "best_params.json").read_text())

    table, X, y, meta = load_features(cfg)
    fit_rows = table["partition"].isin(["train", "val"]).to_numpy()
    print(f"Refitting on train + validation: {fit_rows.sum()} reviews\n")

    model_dir = project_path(cfg["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    times = []
    for name in MODEL_NAMES:
        model = build_model(name, best[name]["params"], seed, n_emb=meta["dim"])
        start = time.perf_counter()
        model.fit(X[fit_rows], y[fit_rows])
        secs = time.perf_counter() - start
        joblib.dump(model, model_dir / f"{name}.joblib")
        times.append({"model": name, "train_seconds": round(secs, 3)})
        print(f"{name:16s} fitted in {secs:6.2f} s  -> models/{name}.joblib")

    times = pd.DataFrame(times)
    times["embedding_ms_per_review"] = times["model"].map({"m2_minilm": meta["embed_ms_per_review"]})
    times.to_csv(project_path(cfg["paths"]["tables"]) / "train_times.csv", index=False)


if __name__ == "__main__":
    main()
