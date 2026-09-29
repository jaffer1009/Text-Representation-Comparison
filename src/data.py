"""Corpus loading, the fixed split protocol, and the H3 test-set slices.

Split protocol (fixed once, here, so no experiment can quietly change it):

    IMDB train (25k) --> 20k fit pool  +  5k dev  (dev selects C, nothing else)
    IMDB test  (25k) --> untouched until final evaluation

The out-of-domain corpora are *only ever* evaluation sets. Nothing is fitted on them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .config import DATA_CACHE, DEV_SIZE, SEED


@dataclass
class Corpus:
    """A labelled text collection. `y` is 0 = negative, 1 = positive throughout."""
    name: str
    texts: list[str]
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.texts)


def _load_hf(path: str, split: str, text_key: str, label_key: str) -> tuple[list[str], np.ndarray]:
    from datasets import load_dataset

    ds = load_dataset(path, split=split, cache_dir=str(DATA_CACHE))
    return list(ds[text_key]), np.asarray(ds[label_key])


def load_imdb() -> tuple[Corpus, Corpus, Corpus]:
    """Return (fit_pool, dev, test) for IMDB with the fixed split above."""
    texts, y = _load_hf("imdb", "train", "text", "label")
    texts, y = np.asarray(texts, dtype=object), np.asarray(y)

    # Stratified dev carve-out, deterministic given SEED.
    rng = np.random.default_rng(SEED)
    dev_idx = np.concatenate([
        rng.permutation(np.flatnonzero(y == c))[: DEV_SIZE // 2] for c in (0, 1)
    ])
    mask = np.zeros(len(y), dtype=bool)
    mask[dev_idx] = True

    fit = Corpus("imdb-fit", list(texts[~mask]), y[~mask])
    dev = Corpus("imdb-dev", list(texts[mask]), y[mask])

    te_texts, te_y = _load_hf("imdb", "test", "text", "label")
    test = Corpus("imdb-test", te_texts, te_y)
    return fit, dev, test


def load_ood() -> list[Corpus]:
    """Out-of-domain evaluation sets for H3. Never trained on.

    - Rotten Tomatoes: same domain (film), drastically shorter texts -> isolates
      a *length* shift from a *topic* shift.
    - Twitter US Airline: different domain, different register, noisy -> full shift.
      Neutral tweets are dropped so the label space matches IMDB's binary one.
    """
    out: list[Corpus] = []

    try:
        texts, y = _load_hf("rotten_tomatoes", "test", "text", "label")
        out.append(Corpus("rotten-tomatoes", texts, y))
    except Exception as exc:  # pragma: no cover - network/dataset availability
        print(f"[data] skipping rotten_tomatoes: {exc}")

    try:
        from datasets import load_dataset

        ds = load_dataset(
            "osanseviero/twitter-airline-sentiment", split="train", cache_dir=str(DATA_CACHE)
        )
        keep = [
            (t, 1 if s == "positive" else 0)
            for t, s in zip(ds["text"], ds["airline_sentiment"])
            if s in ("positive", "negative")
        ]
        rng = np.random.default_rng(SEED)
        # Class-balance it: the raw corpus is ~80% negative, which would make
        # accuracy uninterpretable against IMDB's balanced 50/50.
        neg = [p for p in keep if p[1] == 0]
        pos = [p for p in keep if p[1] == 1]
        n = min(len(neg), len(pos), 1_500)
        sel = [neg[i] for i in rng.permutation(len(neg))[:n]] + \
              [pos[i] for i in rng.permutation(len(pos))[:n]]
        out.append(Corpus("twitter-airline", [t for t, _ in sel],
                          np.asarray([l for _, l in sel])))
    except Exception as exc:  # pragma: no cover
        print(f"[data] skipping twitter-airline: {exc}")

    return out


# --- H3 slices -------------------------------------------------------------
# Applied to the IMDB *test* set only. Each returns a boolean mask.

_NEG_CUE = re.compile(r"\b(not|never|no|nothing|cannot)\b|n't", re.I)
_CONTRAST_CUE = re.compile(r"\b(but|however|although|though|despite|yet)\b", re.I)


def slice_masks(corpus: Corpus) -> dict[str, np.ndarray]:
    """Named boolean masks over `corpus`, for per-slice accuracy in Experiment 3."""
    lengths = np.asarray([len(t.split()) for t in corpus.texts])
    return {
        "negation":    np.asarray([bool(_NEG_CUE.search(t)) for t in corpus.texts]),
        "no-negation": np.asarray([not _NEG_CUE.search(t) for t in corpus.texts]),
        "contrast":    np.asarray([bool(_CONTRAST_CUE.search(t)) for t in corpus.texts]),
        "no-contrast": np.asarray([not _CONTRAST_CUE.search(t) for t in corpus.texts]),
        # 400 words is well past MAX_LEN=256 tokens, so `long` is exactly the set
        # where DistilBERT provably never sees the end of the review.
        "short (<128w)": lengths < 128,
        "long (>400w)":  lengths > 400,
    }


def load_all(smoke: bool = False, with_ood: bool = True):
    """Single entry point used by prepare.py and all three experiments.

    Returns (fit, dev, test, ood). In smoke mode every corpus is cut to 200
    stratified documents and renamed with a `-smoke` suffix, so smoke artifacts
    never collide with the real cache and every caller agrees on the names.
    """
    fit, dev, test = load_imdb()
    ood = load_ood() if with_ood else []
    if not smoke:
        return fit, dev, test, ood

    def shrink(c: Corpus) -> Corpus:
        idx = subsample(min(200, 2 * (len(c) // 2)), c.y, SEED)
        return Corpus(c.name + "-smoke", [c.texts[i] for i in idx], c.y[idx])

    return shrink(fit), shrink(dev), shrink(test), [shrink(c) for c in ood]


def subsample(n: int, y: np.ndarray, seed: int) -> np.ndarray:
    """Stratified subsample of `n` indices, balanced across classes."""
    rng = np.random.default_rng(seed)
    per = n // 2
    idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per] for c in (0, 1)])
    return rng.permutation(idx)
