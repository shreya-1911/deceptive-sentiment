"""Stage 5: error analysis and model interpretation.

Usage (from the project root, after 05_evaluate.py):
    python scripts/06_error_analysis.py

What it does
1. Error types per model and polarity on the TEST set
   (false positive = truthful review flagged as deceptive; false negative = deceptive review missed).
2. Review length of correct vs misclassified test reviews (links to MiniLM truncation).
3. Test reviews misclassified by every learned model, by TF-IDF only, by MiniLM only.
4. The words TF-IDF relies on most: one model per polarity, refitted on train+val with
   Model 1's chosen settings. This is for interpretation only; no test data is used.

Outputs
    results/tables/error_types.csv
    results/tables/error_length.csv
    results/tables/error_examples.csv
    results/tables/top_terms_by_polarity.csv
    results/figures/top_terms_by_polarity.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_config  # noqa: E402
from src.models import MODEL_LABELS, build_model  # noqa: E402
from src.utils import load_features, project_path, set_seed  # noqa: E402

LEARNED = ["m1_tfidf", "m2_minilm", "m3_tfidf_vader"]
TOP_K = 15
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DECEPTIVE_COLOR, TRUTHFUL_COLOR = "#6250d6", "#1baf7a"   # violet / aqua: not reused elsewhere


def error_types(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in LEARNED:
        for pol in ("positive", "negative"):
            s = pred[pred["polarity"] == pol]
            p, y = s[f"pred_{name}"], s["label"]
            rows.append({
                "model": name, "polarity": pol, "n": len(s),
                "false_positives": int(((p == 1) & (y == 0)).sum()),   # truthful flagged deceptive
                "false_negatives": int(((p == 0) & (y == 1)).sum()),   # deceptive missed
                "errors": int((p != y).sum()),
            })
    return pd.DataFrame(rows)


def error_length(pred: pd.DataFrame) -> pd.DataFrame:
    words = pred["text_raw"].str.split().str.len()
    rows = []
    for name in LEARNED:
        wrong = pred[f"pred_{name}"] != pred["label"]
        rows.append({"model": name,
                     "mean_words_correct": round(words[~wrong].mean(), 1),
                     "mean_words_wrong": round(words[wrong].mean(), 1),
                     "n_wrong": int(wrong.sum())})
    return pd.DataFrame(rows)


def error_examples(pred: pd.DataFrame) -> pd.DataFrame:
    wrong = {n: pred[f"pred_{n}"] != pred["label"] for n in LEARNED}
    groups = {
        "all_models_wrong": wrong["m1_tfidf"] & wrong["m2_minilm"] & wrong["m3_tfidf_vader"],
        "tfidf_wrong_minilm_right": wrong["m1_tfidf"] & ~wrong["m2_minilm"],
        "minilm_wrong_tfidf_right": wrong["m2_minilm"] & ~wrong["m1_tfidf"],
    }
    parts = []
    for g, mask in groups.items():
        s = pred[mask].copy()
        s.insert(0, "group", g)
        parts.append(s)
    out = pd.concat(parts)
    cols = ["group", "review_id", "cell", "label"] + [f"prob_{n}" for n in LEARNED] + ["text_raw"]
    return out[cols]


def top_terms(table: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, params: dict, seed: int) -> pd.DataFrame:
    """Refit Model 1 separately on positive and negative train+val reviews; return strongest terms."""
    rows = []
    fit = table["partition"].isin(["train", "val"]).to_numpy()
    for pol in ("positive", "negative"):
        mask = fit & (table["polarity"] == pol).to_numpy()
        model = build_model("m1_tfidf", params, seed).fit(X[mask], y[mask])
        names = model.named_steps["features"].get_feature_names_out()
        names = np.array([n.split("__", 1)[1] for n in names])
        coef = model.named_steps["clf"].coef_[0]          # > 0 pushes towards "deceptive"
        order = np.argsort(coef)
        for direction, idx in (("deceptive", order[::-1][:TOP_K]), ("truthful", order[:TOP_K])):
            for rank, i in enumerate(idx, 1):
                rows.append({"polarity": pol, "points_to": direction, "rank": rank,
                             "term": names[i], "coefficient": round(float(coef[i]), 4)})
    return pd.DataFrame(rows)


def plot_terms(terms: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.4), dpi=300, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    lim = terms["coefficient"].abs().max() * 1.1
    for ax, pol in zip(axes, ("positive", "negative")):
        t = terms[(terms["polarity"] == pol) & (terms["rank"] <= 10)]
        t = pd.concat([t[t["points_to"] == "deceptive"].sort_values("coefficient"),
                       t[t["points_to"] == "truthful"].sort_values("coefficient")])
        t = t.sort_values("coefficient")
        colors = [DECEPTIVE_COLOR if c > 0 else TRUTHFUL_COLOR for c in t["coefficient"]]
        ax.barh(range(len(t)), t["coefficient"], color=colors, height=0.7)
        ax.set_yticks(range(len(t)), t["term"], fontsize=7.5, color=INK)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_facecolor(SURFACE)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK_2, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{pol.capitalize()} reviews", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Logistic regression coefficient", fontsize=8, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DECEPTIVE_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=TRUTHFUL_COLOR)]
    fig.legend(handles, ["Points to deceptive", "Points to truthful"], loc="lower center",
               ncol=2, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle("Strongest TF-IDF terms, separate models per polarity (train + validation)",
                 x=0.02, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    tables, figures = project_path(cfg["paths"]["tables"]), project_path(cfg["paths"]["figures"])
    pred = pd.read_csv(project_path(cfg["paths"]["predictions"]) / "test_predictions.csv")

    et = error_types(pred)
    et.to_csv(tables / "error_types.csv", index=False)
    el = error_length(pred)
    el.to_csv(tables / "error_length.csv", index=False)
    ex = error_examples(pred)
    ex.to_csv(tables / "error_examples.csv", index=False)

    best = json.loads((project_path("results") / "best_params.json").read_text())
    table, X, y, _ = load_features(cfg)
    terms = top_terms(table, X, y, best["m1_tfidf"]["params"], seed)
    terms.to_csv(tables / "top_terms_by_polarity.csv", index=False)
    plot_terms(terms, figures / "top_terms_by_polarity.png")

    print("Error types on the test set (FP = truthful flagged deceptive, FN = deceptive missed)")
    print(et.to_string(index=False), "\n")
    print("Review length (words) of correct vs misclassified test reviews")
    print(el.to_string(index=False), "\n")
    print("Error groups:", ex["group"].value_counts().to_dict(), "\n")
    for pol in ("positive", "negative"):
        for d in ("deceptive", "truthful"):
            t = terms[(terms["polarity"] == pol) & (terms["points_to"] == d)].head(10)
            print(f"{pol:8s} -> {d:9s}: {', '.join(t['term'])}")
    

if __name__ == "__main__":
    main()
