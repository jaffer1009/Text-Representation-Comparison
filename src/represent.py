"""The five representations, behind one interface.

Every representation is a *frozen feature extractor*: it is fitted once on the
IMDB training pool and then only ever `transform`s. The classifier head
(evaluate.py) is identical across all five, so the only thing that varies between
conditions is the representation itself. That is what makes the comparison fair,
and it is also what makes the whole study affordable on CPU: each corpus is
encoded exactly once and cached, after which every experiment is a cheap
logistic-regression fit on the cached matrix.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .config import ARTIFACTS, BERT_MODEL, GLOVE_NAME, MAX_LEN, SEED, W2V_DIM
from .data import Corpus

_TOKEN = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Representation:
    """Fit once on the training pool, then transform (and cache) any corpus."""

    rep_id: str = ""

    def fit(self, corpus: Corpus) -> "Representation":
        raise NotImplementedError

    def _transform(self, corpus: Corpus):
        raise NotImplementedError

    def _is_fitted(self) -> bool:
        """Whether this instance can encode. Subclasses set an attribute in fit()."""
        return any(hasattr(self, a) for a in ("vec", "kv", "model"))

    # -- caching ------------------------------------------------------------
    def _cache_path(self, corpus: Corpus) -> Path:
        return ARTIFACTS / f"{self.rep_id}__{corpus.name}"

    def transform(self, corpus: Corpus, use_cache: bool = True):
        base = self._cache_path(corpus)
        dense, sparse = base.with_suffix(".npy"), base.with_suffix(".npz")
        if use_cache:
            if dense.exists():
                return np.load(dense)
            if sparse.exists():
                return sp.load_npz(sparse)

        if not self._is_fitted():
            raise RuntimeError(
                f"No cached features for '{self.rep_id}' x '{corpus.name}' and the "
                f"representation is not fitted. Run `python -m src.prepare"
                f"{' --smoke' if corpus.name.endswith('-smoke') else ''}` first."
            )

        t0 = time.perf_counter()
        X = self._transform(corpus)
        elapsed = time.perf_counter() - t0
        print(f"[represent] {self.rep_id} x {corpus.name}: {X.shape} in {elapsed:.1f}s")

        if sp.issparse(X):
            sp.save_npz(sparse, X.tocsr())
        else:
            np.save(dense, X)
        base.with_suffix(".json").write_text(
            json.dumps({"encode_seconds": elapsed, "shape": list(X.shape)})
        )
        return X


class TfidfRep(Representation):
    """Sparse lexical baseline. Sees which words occur, never their order."""

    rep_id = "tfidf"

    def fit(self, corpus: Corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),      # bigrams recover a little local order ("not good")
            min_df=2,
            sublinear_tf=True,       # 1+log(tf); raw counts over-weight repetition
            max_features=50_000,
            strip_accents="unicode",
        )
        self.vec.fit(corpus.texts)
        return self

    def _transform(self, corpus: Corpus):
        return self.vec.transform(corpus.texts)


class Word2VecRep(Representation):
    """Static dense vectors, mean-pooled over the document.

    Mean pooling is the standard document representation for static embeddings
    and is deliberately order-blind - like TF-IDF, but in 300 dense dimensions
    instead of 50k sparse ones.
    """

    def __init__(self, pretrained: bool):
        self.pretrained = pretrained
        self.rep_id = "w2v-pre" if pretrained else "w2v-self"

    def fit(self, corpus: Corpus):
        if self.pretrained:
            import gensim.downloader as api

            self.kv = api.load(GLOVE_NAME)
        else:
            from gensim.models import Word2Vec

            sentences = [_tokenize(t) for t in corpus.texts]
            model = Word2Vec(
                sentences,
                vector_size=W2V_DIM,
                window=5,
                min_count=5,
                sg=1,              # skip-gram: better for the rarer sentiment-bearing words
                workers=4,
                epochs=5,
                seed=SEED,
            )
            self.kv = model.wv
        return self

    def _transform(self, corpus: Corpus):
        dim = self.kv.vector_size
        out = np.zeros((len(corpus), dim), dtype=np.float32)
        for i, text in enumerate(corpus.texts):
            vecs = [self.kv[t] for t in _tokenize(text) if t in self.kv]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


class BertRep(Representation):
    """Frozen DistilBERT features. NOT fine-tuned - see README for why.

    `pool='cls'` takes the [CLS] vector; `pool='mean'` takes a mask-aware mean of
    the final layer. Without fine-tuning, [CLS] is known to be a poor sentence
    vector (Reimers & Gurevych, 2019), so carrying both is itself a result.
    """

    def __init__(self, pool: str):
        assert pool in ("cls", "mean")
        self.pool = pool
        self.rep_id = f"bert-{pool}"

    def fit(self, corpus: Corpus):  # nothing is learned; weights are frozen
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4)))
        self.tok = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.model = AutoModel.from_pretrained(BERT_MODEL).eval()
        return self

    def _encode_both(self, corpus: Corpus, batch_size: int = 32):
        """One forward pass, both poolings. Returns (cls, mean).

        The two poolings are different reads of the *same* hidden states, so
        running the model twice would double the only expensive step in the
        project for no information gain.

        Texts are processed in length order so that each batch pads to its own
        longest member rather than to the global maximum; results are restored to
        the original order before returning.
        """
        import torch
        from tqdm import tqdm

        n = len(corpus)
        order = np.argsort([len(t) for t in corpus.texts])
        cls = np.zeros((n, 768), dtype=np.float32)
        mean = np.zeros((n, 768), dtype=np.float32)

        with torch.inference_mode():
            for i in tqdm(range(0, n, batch_size),
                          desc=f"bert/{corpus.name}", unit="batch"):
                sel = order[i : i + batch_size]
                enc = self.tok([corpus.texts[j] for j in sel], truncation=True,
                               max_length=MAX_LEN, padding=True, return_tensors="pt")
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                cls[sel] = hidden[:, 0].numpy().astype(np.float32)
                mean[sel] = ((hidden * mask).sum(1) /
                             mask.sum(1).clamp(min=1e-9)).numpy().astype(np.float32)
        return cls, mean

    def _transform(self, corpus: Corpus):
        cls, mean = self._encode_both(corpus)
        return cls if self.pool == "cls" else mean


def encode_bert_both(fit_corpus: Corpus, corpora: list[Corpus]) -> float:
    """Encode every corpus once, writing both the bert-cls and bert-mean caches.

    Returns the model-load time. This exists so the single expensive pass over
    the data serves both BERT conditions.
    """
    t0 = time.perf_counter()
    enc = BertRep("mean").fit(fit_corpus)
    load_s = time.perf_counter() - t0

    for c in corpora:
        paths = {p: ARTIFACTS / f"bert-{p}__{c.name}.npy" for p in ("cls", "mean")}
        if all(p.exists() for p in paths.values()):
            print(f"[represent] bert x {c.name}: cached, skipping")
            continue

        t1 = time.perf_counter()
        cls, mean = enc._encode_both(c)
        elapsed = time.perf_counter() - t1
        np.save(paths["cls"], cls)
        np.save(paths["mean"], mean)
        for pool in ("cls", "mean"):
            (ARTIFACTS / f"bert-{pool}__{c.name}.json").write_text(
                json.dumps({"encode_seconds": elapsed, "shape": list(cls.shape),
                            "note": "one shared forward pass produced both poolings"})
            )
        print(f"[represent] bert x {c.name}: {cls.shape} in {elapsed:.1f}s (both poolings)")
    return load_s


def build(rep_id: str) -> Representation:
    return {
        "tfidf": lambda: TfidfRep(),
        "w2v-self": lambda: Word2VecRep(pretrained=False),
        "w2v-pre": lambda: Word2VecRep(pretrained=True),
        "bert-cls": lambda: BertRep("cls"),
        "bert-mean": lambda: BertRep("mean"),
    }[rep_id]()
