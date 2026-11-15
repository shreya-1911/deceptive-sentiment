"""Feature extraction: TF-IDF, MiniLM sentence embeddings, VADER sentiment scores.

Only TF-IDF and the VADER scaler are *learned* from data, and they live inside
scikit-learn Pipelines (see models.py), so they are fitted on training data only.
MiniLM and VADER are pretrained/fixed, so computing them once for every review and
caching the result leaks nothing: no statistic from validation or test data is used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

VADER_COLS = ["vader_neg", "vader_neu", "vader_pos", "vader_compound", "vader_abs_compound"]


# ---------------------------------------------------------------- TF-IDF
def make_tfidf(min_df: int = 1, sublinear_tf: bool = False) -> TfidfVectorizer:
    """Unigram + bigram TF-IDF. Lowercases, but keeps stop words, negations and pronouns."""
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        stop_words=None,
        token_pattern=r"(?u)\b\w+\b",   # also keeps 1-letter tokens such as "I" and "a"
    )


# ---------------------------------------------------------------- MiniLM
def _cache_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def get_embeddings(df: pd.DataFrame, model_name: str, cache_dir: Path,
                   batch_size: int = 32) -> tuple[np.ndarray, dict]:
    """Return L2-normalised embeddings aligned row-for-row with `df`, plus metadata.

    Embeddings are cached by review_id under cache_dir. Metadata records the
    embedding time and how many reviews exceeded the model's input limit.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = _cache_name(model_name)
    vec_path, meta_path = cache_dir / f"{name}.npz", cache_dir / f"{name}_meta.json"

    if vec_path.exists():
        z = np.load(vec_path)
        lookup = dict(zip(z["review_id"], range(len(z["review_id"]))))
        if all(r in lookup for r in df["review_id"]):
            meta = json.loads(meta_path.read_text())
            return z["emb"][[lookup[r] for r in df["review_id"]]], meta

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    texts = df["text_clean"].tolist()

    tok_lens = [len(model.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    truncated = np.array(tok_lens) > model.max_seq_length

    start = time.perf_counter()
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    seconds = time.perf_counter() - start

    np.savez(vec_path, review_id=df["review_id"].to_numpy(), emb=emb)
    meta = {
        "model": model_name,
        "dim": int(emb.shape[1]),
        "max_seq_length": int(model.max_seq_length),
        "n_reviews": len(texts),
        "embed_seconds_total": round(seconds, 3),
        "embed_ms_per_review": round(1000 * seconds / len(texts), 3),
        "n_truncated": int(truncated.sum()),
        "truncated_by_cell": (pd.Series(truncated, index=df["cell"].to_numpy())
                              .groupby(level=0).agg(["sum", "count"])
                              .astype(int).to_dict(orient="index")),
        "mean_tokens_by_cell": (pd.Series(tok_lens, index=df["cell"].to_numpy())
                                .groupby(level=0).mean().round(1).to_dict()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return emb, meta


# ---------------------------------------------------------------- VADER
def _vader():
    # vaderSentiment is the reference implementation (Hutto & Gilbert, 2014);
    # its lexicon ships inside the package, so no separate download is needed.
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def get_vader(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """VADER scores per review (computed on cleaned, NOT lowercased text: VADER uses case and '!')."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "vader.csv"
    if path.exists():
        cached = pd.read_csv(path).set_index("review_id")
        if df["review_id"].isin(cached.index).all():
            return cached.loc[df["review_id"], VADER_COLS].reset_index(drop=True)

    sia = _vader()
    scores = pd.DataFrame([sia.polarity_scores(t) for t in df["text_clean"]])
    out = pd.DataFrame({
        "review_id": df["review_id"].to_numpy(),
        "vader_neg": scores["neg"], "vader_neu": scores["neu"], "vader_pos": scores["pos"],
        "vader_compound": scores["compound"], "vader_abs_compound": scores["compound"].abs(),
    })
    out.to_csv(path, index=False)
    return out[VADER_COLS].reset_index(drop=True)
