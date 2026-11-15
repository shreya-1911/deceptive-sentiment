"""Reproducibility and leakage checks. Run with:  pytest

These tests use the real dataset and split file, so run scripts 01 and 02 first.
They are skipped (not failed) if the data has not been downloaded.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data import LEAKY_COLUMNS, load_config, load_dataset, normalise_for_dedup

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
RAW = ROOT / CFG["paths"]["raw_data"]
SPLITS = ROOT / CFG["paths"]["splits"]

needs_data = pytest.mark.skipif(not RAW.exists(), reason="dataset not downloaded")
needs_splits = pytest.mark.skipif(not SPLITS.exists(), reason="run scripts/02_make_splits.py first")


@needs_data
def test_leaky_column_removed():
    df = load_dataset(RAW)
    assert not set(LEAKY_COLUMNS) & set(df.columns)


@needs_data
def test_review_ids_unique():
    df = load_dataset(RAW)
    assert df["review_id"].is_unique


@needs_data
@needs_splits
def test_each_review_in_at_most_one_partition():
    splits = pd.read_csv(SPLITS)
    assert splits["review_id"].is_unique


@needs_data
@needs_splits
def test_no_text_shared_across_partitions():
    df = load_dataset(RAW).merge(pd.read_csv(SPLITS)[["review_id", "partition"]], on="review_id")
    df["key"] = df["text_raw"].map(normalise_for_dedup)
    assert (df.groupby("key")["partition"].nunique() == 1).all()


@needs_data
@needs_splits
def test_partitions_stratified():
    splits = pd.read_csv(SPLITS)
    shares = pd.crosstab(splits["partition"], splits["cell"], normalize="index")
    assert (shares.sub(0.25).abs() < 0.02).all().all(), shares
