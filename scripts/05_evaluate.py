"""Stage 4: final evaluation on the TEST set (run once, after tuning is finished).

Usage (from the project root):
    python scripts/05_evaluate.py

Outputs
    results/predictions/test_predictions.csv   one row per test review, every model's prediction
    results/tables/test_metrics.csv            all / positive / negative, with 95% bootstrap CIs
    results/tables/test_metrics_report.csv     same, formatted "0.86 [0.81, 0.90]" for the report
    results/tables/polarity_difference.csv     H2: macro-F1(positive) - macro-F1(negative), with CI
    results/tables/mcnemar.csv                 H3/H4: paired model comparisons
    results/tables/inference_times.csv
    results/tables/classification_reports.txt  per-class precision/recall/F1
    results/figures/macro_f1_by_polarity.png
    results/figures/confusion_matrices.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_config  # noqa: E402
from src.evaluate import bootstrap_ci, bootstrap_diff, mcnemar, point_metrics  # noqa: E402
from src.models import MODEL_LABELS, MODEL_NAMES  # noqa: E402
from src.utils import load_features, project_path, set_seed  # noqa: E402

SUBSETS = {"all": None, "positive": "positive", "negative": "negative"}
LEARNED = ["m1_tfidf", "m2_minilm", "m3_tfidf_vader"]
COMPARISONS = [("m1_tfidf", "m2_minilm"), ("m1_tfidf", "m3_tfidf_vader"), ("m2_minilm", "m3_tfidf_vader")]

# Chart styling: one categorical colour per subset, plus distinct marker shapes so
# the chart still reads in greyscale print and for colour-blind readers.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SUBSET_STYLE = {
    "all": dict(color=INK_2, marker="o", label="All test reviews"),
    "positive": dict(color="#2a78d6", marker="s", label="Positive reviews"),
    "negative": dict(color="#eb6834", marker="^", label="Negative reviews"),
}
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#86b6ef", "#2a78d6", "#184f95"])


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_macro_f1(metrics: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    offsets = {"all": -0.22, "positive": 0.0, "negative": 0.22}
    for subset, st in SUBSET_STYLE.items():
        d = metrics[metrics["subset"] == subset].set_index("model").loc[MODEL_NAMES]
        x = np.arange(len(MODEL_NAMES)) + offsets[subset]
        err = np.vstack([d["macro_f1"] - d["macro_f1_lo"], d["macro_f1_hi"] - d["macro_f1"]])
        ax.errorbar(x, d["macro_f1"], yerr=err, fmt=st["marker"], color=st["color"],
                    markersize=6, elinewidth=1.6, capsize=0, label=st["label"], zorder=3)
    ax.set_xticks(range(len(MODEL_NAMES)), [MODEL_LABELS[m] for m in MODEL_NAMES], color=INK)
    ax.set_ylim(0, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Macro-F1 (test set)", color=INK_2, fontsize=9)
    ax.set_title("Macro-F1 by model and review polarity, with 95% bootstrap intervals",
                 loc="left", fontsize=10, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_confusions(pred: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    vmax = max(len(pred[pred["polarity"] == p]) // 2 for p in ("positive", "negative"))
    for r, pol in enumerate(("positive", "negative")):
        sub = pred[pred["polarity"] == pol]
        for c, name in enumerate(LEARNED):
            ax = axes[r, c]
            cm = confusion_matrix(sub["label"], sub[f"pred_{name}"], labels=[0, 1])
            ax.imshow(cm, cmap=SEQ_BLUE, vmin=0, vmax=vmax)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                            color="white" if cm[i, j] > 0.6 * vmax else INK)
            ax.set_xticks([0, 1], ["truthful", "deceptive"], fontsize=8, color=INK_2)
            ax.set_yticks([0, 1], ["truthful", "deceptive"], fontsize=8, color=INK_2)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(MODEL_LABELS[name], fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(f"{pol.capitalize()} reviews\nTrue label", fontsize=8, color=INK_2)
            if r == 1:
                ax.set_xlabel("Predicted", fontsize=8, color=INK_2)
    fig.suptitle("Test-set confusion matrices by review polarity", x=0.02, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    ev = cfg["evaluation"]
    n_boot, level = ev["bootstrap_resamples"], ev["ci_level"]
    tables, figures = project_path(cfg["paths"]["tables"]), project_path(cfg["paths"]["figures"])
    preds_dir = project_path(cfg["paths"]["predictions"])
    for d in (tables, figures, preds_dir):
        d.mkdir(parents=True, exist_ok=True)

    table, X, y, _ = load_features(cfg)
    te = (table["partition"] == "test").to_numpy()
    pred = table.loc[te, ["review_id", "hotel", "polarity", "cell", "label", "text_raw"]].reset_index(drop=True)
    print(f"Test reviews: {te.sum()}\n")

    # --- predictions and inference time
    times = []
    for name in MODEL_NAMES:
        model = joblib.load(project_path(cfg["paths"]["models"]) / f"{name}.joblib")
        start = time.perf_counter()
        pred[f"pred_{name}"] = model.predict(X[te])
        secs = time.perf_counter() - start
        pred[f"prob_{name}"] = model.predict_proba(X[te])[:, 1].round(4)
        times.append({"model": name, "inference_seconds": round(secs, 4),
                      "inference_ms_per_review": round(1000 * secs / te.sum(), 4)})
    pd.DataFrame(times).to_csv(tables / "inference_times.csv", index=False)
    pred.to_csv(preds_dir / "test_predictions.csv", index=False)

    # --- metrics with bootstrap CIs, per subset
    rows = []
    for name in MODEL_NAMES:
        for subset, pol in SUBSETS.items():
            s = pred if pol is None else pred[pred["polarity"] == pol]
            yt, yp = s["label"].to_numpy(), s[f"pred_{name}"].to_numpy()
            rows.append({"model": name, "subset": subset, "n": len(s),
                         **point_metrics(yt, yp), **bootstrap_ci(yt, yp, n_boot, level, seed)})
    metrics = pd.DataFrame(rows)
    metrics.round(4).to_csv(tables / "test_metrics.csv", index=False)

    fmt = metrics[["model", "subset", "n"]].copy()
    for k in ("accuracy", "precision", "recall", "macro_f1"):
        fmt[k] = [f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
                  for v, lo, hi in zip(metrics[k], metrics[f"{k}_lo"], metrics[f"{k}_hi"])]
    fmt.to_csv(tables / "test_metrics_report.csv", index=False)

    # --- H2: positive minus negative, per model
    pos, neg = pred[pred["polarity"] == "positive"], pred[pred["polarity"] == "negative"]
    h2 = pd.DataFrame([{"model": name, **bootstrap_diff(
        pos["label"], pos[f"pred_{name}"], neg["label"], neg[f"pred_{name}"],
        "macro_f1", n_boot, level, seed)} for name in LEARNED])
    h2["ci_excludes_zero"] = (h2["diff_lo"] > 0) | (h2["diff_hi"] < 0)
    h2.round(4).to_csv(tables / "polarity_difference.csv", index=False)

    # --- H3 / H4: McNemar on paired predictions
    mc = []
    for a, b in COMPARISONS:
        for subset, pol in SUBSETS.items():
            s = pred if pol is None else pred[pred["polarity"] == pol]
            mc.append({"model_a": a, "model_b": b, "subset": subset,
                       **mcnemar(s["label"], s[f"pred_{a}"], s[f"pred_{b}"])})
    pd.DataFrame(mc).round(4).to_csv(tables / "mcnemar.csv", index=False)

    # --- per-class reports
    with open(tables / "classification_reports.txt", "w", encoding="utf-8") as f:
        for name in MODEL_NAMES:
            for subset, pol in SUBSETS.items():
                s = pred if pol is None else pred[pred["polarity"] == pol]
                f.write(f"=== {name} | {subset} (n={len(s)}) ===\n")
                f.write(classification_report(s["label"], s[f"pred_{name}"], labels=[0, 1],
                                              target_names=["truthful", "deceptive"],
                                              digits=3, zero_division=0))
                f.write("\n")

    plot_macro_f1(metrics, figures / "macro_f1_by_polarity.png")
    plot_confusions(pred, figures / "confusion_matrices.png")

    # --- console summary
    show = fmt.pivot(index="model", columns="subset", values="macro_f1").loc[MODEL_NAMES, list(SUBSETS)]
    print("Macro-F1 [95% CI]")
    print(show.to_string(), "\n")
    print("H2  positive - negative macro-F1")
    print(h2.round(3).to_string(index=False), "\n")
    print("McNemar (full test set)")
    print(pd.DataFrame(mc).query("subset == 'all'").round(4).to_string(index=False))
    


if __name__ == "__main__":
    main()
