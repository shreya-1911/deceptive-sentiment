"""Loading, validation and light cleaning of the Deceptive Opinion Spam Corpus.

Source: Ott et al. (2011; 2013), via Kaggle file `deceptive-opinion.csv`.
Columns expected: deceptive, hotel, polarity, source, text.

Design notes
- `source` is dropped on load: every deceptive review comes from Mechanical Turk,
  so it predicts the label perfectly (label leakage).
- `text_raw` keeps the original text untouched. `text_clean` only normalises
  whitespace and stray markup; it does NOT lowercase, remove stop words or
  lemmatise, so negations and pronouns survive. Model-specific steps
  (e.g. lowercasing for TF-IDF) happen inside each model's pipeline.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["deceptive", "hotel", "polarity", "source", "text"]
LEAKY_COLUMNS = ["source"]
LABEL_VALUES = {"deceptive": {"truthful", "deceptive"},
                "polarity": {"positive", "negative"}}


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV exactly as distributed and check its structure."""
    df = pd.read_csv(path, encoding="utf-8")
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    for col, allowed in LABEL_VALUES.items():
        found = set(df[col].dropna().str.strip().str.lower().unique())
        if not found <= allowed:
            raise ValueError(f"Unexpected values in '{col}': {sorted(found - allowed)}")
    return df


def clean_text(text: str) -> str:
    """Minimal, meaning-preserving cleaning: unescape HTML, strip tags, collapse whitespace."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalise_for_dedup(text: str) -> str:
    """Aggressive normalisation used ONLY to detect near-duplicates, never as model input."""
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Return the modelling table: leak-free, with stable ids and binary labels.

    Columns: review_id, hotel, polarity, deceptive, label (1 = deceptive),
    text_raw, text_clean, cell (e.g. 'deceptive_positive').
    """
    df = read_raw(path).copy()
    df["deceptive"] = df["deceptive"].str.strip().str.lower()
    df["polarity"] = df["polarity"].str.strip().str.lower()
    df = df.drop(columns=LEAKY_COLUMNS)

    df.insert(0, "review_id", range(len(df)))  # row order of the original file
    df = df.rename(columns={"text": "text_raw"})
    df["text_clean"] = df["text_raw"].map(clean_text)
    df["label"] = (df["deceptive"] == "deceptive").astype(int)
    df["cell"] = df["deceptive"] + "_" + df["polarity"]

    assert not set(LEAKY_COLUMNS) & set(df.columns), "leaky column survived"
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the first copy (lowest review_id) of each exactly repeated text.

    Returns (kept, dropped). Duplicates are detected on the normalised text, so
    copies that differ only in case, punctuation or spacing are also caught.
    """
    key = df["text_raw"].map(normalise_for_dedup)
    is_dup = key.duplicated(keep="first")
    return df[~is_dup].copy(), df[is_dup].copy()


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml from the project root (or a given path)."""
    import yaml

    path = Path(path) if path else Path(__file__).resolve().parents[1] / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
