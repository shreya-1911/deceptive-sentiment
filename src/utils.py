"""Small shared helpers: seeding, timing, the modelling table."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_config, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Fix every random source we use, so reruns give identical results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def project_path(rel: str) -> Path:
    return ROOT / rel


def load_split_table(cfg: dict | None = None) -> pd.DataFrame:
    """Dataset joined to the saved split; rows not in the split (dropped duplicates) are excluded."""
    cfg = cfg or load_config()
    df = load_dataset(project_path(cfg["paths"]["raw_data"]))
    splits = pd.read_csv(project_path(cfg["paths"]["splits"]))[["review_id", "partition"]]
    out = df.merge(splits, on="review_id", how="inner", validate="one_to_one")
    return out.sort_values("review_id").reset_index(drop=True)


def load_features(cfg: dict):
    """Return (table, X, y, embedding_meta) for every review in the split.

    First run downloads MiniLM (~90 MB) and the VADER lexicon, then caches
    features in data/cache/ so later runs take seconds.
    """
    from src.features import get_embeddings, get_vader
    from src.models import build_frame

    table = load_split_table(cfg)
    cache = project_path(cfg["paths"]["cache"])
    emb, meta = get_embeddings(table, cfg["embedding_model"], cache, cfg.get("embedding_batch_size", 32))
    vader = get_vader(table, cache)
    X = build_frame(table, emb, vader)
    return table, X, table["label"].to_numpy(), meta
