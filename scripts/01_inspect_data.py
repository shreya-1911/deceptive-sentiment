"""Stage 1: inspect the raw dataset and write a reproducible inspection report.

Usage (from the project root):
    python scripts/01_inspect_data.py --data data/raw/deceptive-opinion.csv

Outputs (results/tables/):
    inspection_summary.txt   human-readable report of every check
    cell_counts.csv          deceptive x polarity counts
    length_by_cell.csv       word-count statistics per cell
    duplicates.csv           exact / near-duplicate groups (empty if none)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset, normalise_for_dedup, read_raw  # noqa: E402


def main(data_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731

    raw = read_raw(data_path)
    say(f"File: {data_path}")
    say(f"Rows: {len(raw)}   Columns: {list(raw.columns)}")

    say("\n== Missing values per column ==")
    say(raw.isna().sum().to_string())
    empty_text = (raw["text"].fillna("").str.strip() == "").sum()
    say(f"Empty/whitespace-only texts: {empty_text}")

    say("\n== Leakage check: source vs deceptive (source is dropped afterwards) ==")
    say(pd.crosstab(raw["source"], raw["deceptive"]).to_string())

    df = load_dataset(data_path)

    say("\n== Cell counts (deceptive x polarity) ==")
    cells = pd.crosstab(df["deceptive"], df["polarity"], margins=True)
    say(cells.to_string())
    cells.to_csv(out_dir / "cell_counts.csv")

    say("\n== Reviews per hotel per cell (min / max) ==")
    per_hotel = df.groupby(["hotel", "cell"]).size().unstack(fill_value=0)
    say(f"Hotels: {df['hotel'].nunique()}")
    say(per_hotel.agg(["min", "max"]).to_string())

    say("\n== Word count by cell ==")
    df["n_words"] = df["text_clean"].str.split().str.len()
    lengths = df.groupby("cell")["n_words"].describe().round(1)
    say(lengths.to_string())
    lengths.to_csv(out_dir / "length_by_cell.csv")

    say("\n== Duplicates ==")
    exact = df[df.duplicated("text_raw", keep=False)]
    df["norm"] = df["text_raw"].map(normalise_for_dedup)
    near = df[df.duplicated("norm", keep=False)]
    say(f"Exact duplicate rows: {len(exact)}")
    say(f"Near-duplicate rows (case/punctuation/whitespace-insensitive): {len(near)}")
    if len(near):
        conflicting = near.groupby("norm")["label"].nunique().gt(1).sum()
        say(f"Near-duplicate groups with conflicting labels: {conflicting}")
    near.sort_values("norm")[["review_id", "hotel", "cell", "text_raw"]].to_csv(
        out_dir / "duplicates.csv", index=False)

    (out_dir / "inspection_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data/raw/deceptive-opinion.csv")
    p.add_argument("--out", type=Path, default=ROOT / "results/tables")
    args = p.parse_args()
    main(args.data, args.out)
