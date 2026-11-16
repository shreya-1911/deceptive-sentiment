# Investigating the Effect of Sentiment Polarity on AI-Based Fake Review Detection

Final-year BSc Computer Science with AI — Advanced AI Applications coursework.

Extends Alawadh et al. (2023) by evaluating frequency-based (TF-IDF) and semantic
(sentence-embedding) deception classifiers separately on positive and negative hotel
reviews, and by testing whether lexicon-based sentiment features (VADER) add information.

## Setup

Python 3.10–3.12.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\activate
# Windows Git Bash:   source .venv/Scripts/activate
# macOS/Linux:        source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build
pip install -r requirements.txt
```

On Windows, keep the project in a short path outside OneDrive (e.g. `C:\Projects\`)
to avoid the 260-character path limit during installation.

The first run of `03_tune.py` downloads the MiniLM model (about 90 MB) from
Hugging Face, so an internet connection is needed once. Embeddings and VADER
scores are then cached in `data/cache/`.

## Data

## Data

The dataset is included, so no separate download is needed:
`data/raw/deceptive-opinion.csv`, the Deceptive Opinion Spam Corpus v1.4
(Ott et al., 2011; 2013), as downloaded unchanged from Kaggle:
https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus

It is licensed under **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike), as
stated on the Kaggle page, and is redistributed here under the same licence for
non-commercial academic use. Full attribution: `data/raw/LICENSE.md`. The same licence
applies to result files that contain review text (per-review predictions, error
examples and duplicate listings).

The `source` column is dropped on load because it reveals the label (all deceptive
reviews come from Mechanical Turk).

Trained models are included in `models/` (scikit-learn pipelines saved with joblib),
so `05_evaluate.py` can be rerun directly. `04_train_final.py` regenerates them.

## Run order

Run everything with `python run_all.py`, or step by step:

```bash
python scripts/01_inspect_data.py    # data checks -> results/tables/
python scripts/02_make_splits.py     # fixed 70/15/15 stratified split (seed 42)
python scripts/03_tune.py            # hyperparameters chosen on validation set only
python scripts/04_train_final.py     # refit on train+val, save models/*.joblib
python scripts/05_evaluate.py        # test metrics, CIs, McNemar, figures
python scripts/06_error_analysis.py  # error types, misclassified examples, top TF-IDF terms
python scripts/07_robustness.py      # repeated CV + cross-polarity transfer (train+val only)
python -m pytest                     # leakage and split checks
```

Main outputs: `results/tables/test_metrics_report.csv`,
`results/tables/polarity_difference.csv`, `results/tables/mcnemar.csv`,
`results/figures/macro_f1_by_polarity.png`, `results/figures/confusion_matrices.png`.

## Hypotheses (stated before test results were seen)

- H1: Models 1 and 2 beat the majority baseline.
- H2: TF-IDF macro-F1 is higher on positive than negative reviews.
- H3: TF-IDF and MiniLM differ in macro-F1 (two-sided).
- H4: Adding VADER features changes TF-IDF macro-F1 (two-sided).

## Models

| # | Features | Classifier |
|---|----------|------------|
| 0 | none | majority class |
| 1 | TF-IDF unigrams + bigrams | logistic regression |
| 2 | `sentence-transformers/all-MiniLM-L6-v2` embeddings | logistic regression |
| 3 | TF-IDF + VADER scores | logistic regression |


## References

- Alawadh, H.M., Alabrah, A., Meraj, T. and Rauf, H.T. (2023) 'Semantic features-based discourse analysis using deceptive and real text reviews', *Information*, 14(1), 34. https://doi.org/10.3390/info14010034
- Ott, M., Choi, Y., Cardie, C. and Hancock, J.T. (2011) 'Finding deceptive opinion spam by any stretch of the imagination', *Proc. ACL-HLT 2011*, pp. 309–319.
- Hutto, C.J. and Gilbert, E. (2014) 'VADER: A parsimonious rule-based model for sentiment analysis of social media text', *Proc. ICWSM 2014*. (Used via the `vaderSentiment` package.)
- Pretrained model: `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face). Cite Reimers and Gurevych (2019) for Sentence-BERT and Wang et al. (2020) for MiniLM; verify full details before final submission.
- Ott, M., Cardie, C. and Hancock, J.T. (2013) 'Negative deceptive opinion spam', *Proc. NAACL-HLT 2013*, pp. 497–501.
