"""Shared constants: paths, seeds, the representation grid, and the poster palette.

Every experiment imports from here so that the poster, the figures and the CSVs
cannot drift apart.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"      # cached embeddings (.npy) - gitignored
DATA_CACHE = ROOT / "data_cache"    # downloaded corpora - gitignored
RESULTS = ROOT / "results"          # CSVs behind every poster claim - committed
FIGURES = ROOT / "figures"          # vector PDFs used on the poster - committed

for _d in (ARTIFACTS, DATA_CACHE, RESULTS, FIGURES):
    _d.mkdir(exist_ok=True)

SEED = 42
N_SEEDS = 5                         # subsample seeds for the learning curves
SEEDS = [SEED + i for i in range(N_SEEDS)]

# Held out of the IMDB train split to select the regularisation strength C.
# The 25k test split is touched exactly once, at final evaluation.
DEV_SIZE = 5_000

# Experiment 1 label budgets. Log-spaced: the interesting behaviour is at the
# small end, where a linear grid would waste most of its points.
TRAIN_SIZES = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 20_000]

MAX_LEN = 256                       # DistilBERT truncation window
BERT_MODEL = "distilbert-base-uncased"
W2V_DIM = 300
GLOVE_NAME = "glove-wiki-gigaword-300"

# The five conditions. `family` is what the poster collapses them to; the
# appendix carries the full grid.
REPRESENTATIONS = {
    "tfidf":     {"family": "TF-IDF",   "label": "TF-IDF (1-2 gram)",   "sparse": True},
    "w2v-self":  {"family": "Word2Vec", "label": "Word2Vec (in-domain)", "sparse": False},
    "w2v-pre":   {"family": "Word2Vec", "label": "GloVe (pretrained)",   "sparse": False},
    "bert-cls":  {"family": "BERT",     "label": "DistilBERT [CLS]",     "sparse": False},
    "bert-mean": {"family": "BERT",     "label": "DistilBERT mean-pool", "sparse": False},
}

# One colour per family, used identically in every figure and on the poster, so a
# reader can track a method across the whole poster by colour alone.
FAMILY_COLORS = {
    "TF-IDF":   "#1f6f8b",   # deep teal
    "Word2Vec": "#e08a3c",   # amber
    "BERT":     "#8b3a62",   # plum
}
# Within a family the pretrained/mean variant is the solid line, the other dashed.
REP_STYLE = {
    "tfidf":     {"color": FAMILY_COLORS["TF-IDF"],   "ls": "-"},
    "w2v-self":  {"color": FAMILY_COLORS["Word2Vec"], "ls": "--"},
    "w2v-pre":   {"color": FAMILY_COLORS["Word2Vec"], "ls": "-"},
    "bert-cls":  {"color": FAMILY_COLORS["BERT"],     "ls": "--"},
    "bert-mean": {"color": FAMILY_COLORS["BERT"],     "ls": "-"},
}
