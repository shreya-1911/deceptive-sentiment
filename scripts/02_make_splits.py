"""Stage 1b: create the fixed, stratified train / validation / test partitions.

Usage (from the project root):
    python scripts/02_make_splits.py

Method
- Stratify on the joint label (deceptive x polarity) so every partition keeps
  the same share of all four cells.
- Split is made on unique texts only, so an exact duplicate can never sit in
  both training and test data.
- config `split.drop_duplicates`:
    true  -> extra copies are excluded from all partitions
    false -> extra copies are kept and placed in the same partition as their first copy

Outputs
    data/splits/split_seed42.csv          review_id, cell, partition   (commit this)
    results/tables/split_counts.csv       rows per partition x cell
    results/tables/dropped_duplicates.csv review_ids excluded (if any)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import drop_exact_duplicates, load_config, load_dataset, normalise_for_dedup  # noqa: E402


def make_splits(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = cfg["seed"]
    s = cfg["split"]
    assert abs(s["train"] + s["val"] + s["test"] - 1) < 1e-9, "split ratios must sum to 1"

    unique, dupes = drop_exact_duplicates(df)

    # 1st cut: train vs (val + test); 2nd cut: val vs test.
    train, rest = train_test_split(
        unique, test_size=s["val"] + s["test"], stratify=unique["cell"], random_state=seed)
    val, test = train_test_split(
        rest, test_size=s["test"] / (s["val"] + s["test"]), stratify=rest["cell"], random_state=seed)

    parts = pd.concat([
        train.assign(partition="train"),
        val.assign(partition="val"),
        test.assign(partition="test"),
    ])[["review_id", "cell", "partition"]]

    if not s.get("drop_duplicates", True) and len(dupes):
        # give each extra copy the partition of its first copy
        key_to_part = dict(zip(
            df.set_index("review_id").loc[parts["review_id"], "text_raw"].map(normalise_for_dedup),
            parts["partition"]))
        extra = dupes.assign(partition=dupes["text_raw"].map(normalise_for_dedup).map(key_to_part))
        parts = pd.concat([parts, extra[["review_id", "cell", "partition"]]])
        dupes = dupes.iloc[0:0]

    return parts.sort_values("review_id").reset_index(drop=True), dupes


def main() -> None:
    cfg = load_config()
    p = cfg["paths"]
    df = load_dataset(ROOT / p["raw_data"])

    splits, dropped = make_splits(df, cfg)

    out_split = ROOT / p["splits"]
    out_split.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_split, index=False)

    tables = ROOT / p["tables"]
    tables.mkdir(parents=True, exist_ok=True)
    counts = pd.crosstab(splits["partition"], splits["cell"], margins=True)
    counts = counts.reindex(["train", "val", "test", "All"])
    counts.to_csv(tables / "split_counts.csv")
    dropped[["review_id", "hotel", "cell"]].to_csv(tables / "dropped_duplicates.csv", index=False)

    print(f"Rows loaded: {len(df)}   Duplicates excluded: {len(dropped)}   Rows split: {len(splits)}")
    print(f"Seed: {cfg['seed']}   drop_duplicates: {cfg['split'].get('drop_duplicates', True)}\n")
    print(counts.to_string())
    print(f"\nShares per partition: "
          + ", ".join(f"{k} {v:.1%}" for k, v in splits['partition'].value_counts(normalize=True).items()))
    print(f"\nSplit file written to {out_split}")


if __name__ == "__main__":
    main()
