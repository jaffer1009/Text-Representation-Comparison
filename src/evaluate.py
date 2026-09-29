"""The classifier head, held identical across every condition.

Nothing in this module knows which representation it is handed. That is the
point: any accuracy difference between conditions is attributable to the
representation and to nothing else.

The head is a sklearn Pipeline so that feature scaling travels *with* the fitted
model. Returning a bare classifier would make it possible to score unscaled
features by accident, which fails silently rather than loudly.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED

# Selected on the dev split only. Wide because the sparse and dense conditions
# sit at very different scales.
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def _make_head(C: float, sparse: bool, seed: int) -> Pipeline:
    """Scaler (dense only) + logistic regression.

    Sparse TF-IDF is already L2-normalised, and centring it would densify 50k
    columns into an unusable matrix - hence `with_mean=False` there.
    """
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    if sparse:
        return Pipeline([("clf", clf)])
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def fit_head(Xtr, ytr, Xdev, ydev, seed: int = SEED) -> tuple[Pipeline, float]:
    """Fit the head, choosing C on the dev split. Returns (pipeline, best_C)."""
    sparse = sp.issparse(Xtr)

    best, best_acc = C_GRID[0], -1.0
    for C in C_GRID:
        pipe = _make_head(C, sparse, seed).fit(Xtr, ytr)
        acc = accuracy_score(ydev, pipe.predict(Xdev))
        if acc > best_acc:
            best, best_acc = C, acc

    return _make_head(best, sparse, seed).fit(Xtr, ytr), best


def fit_and_score(Xtr, ytr, Xdev, ydev, eval_sets, seed: int = SEED):
    """Fit once, then score on every (name, X, y) in `eval_sets`.

    Returns (metrics_by_name, best_C, predictions_by_name). Predictions are
    returned so Experiment 3 can slice them without refitting.
    """
    pipe, best_C = fit_head(Xtr, ytr, Xdev, ydev, seed)

    metrics, preds = {}, {}
    for name, X, y in eval_sets:
        p = pipe.predict(X)
        preds[name] = p
        metrics[name] = {
            "accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
        }
    return metrics, best_C, preds


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = SEED):
    """Percentile bootstrap 95% CI for accuracy on a single evaluation set.

    Used for the per-slice bars in Experiment 3, where some slices are small
    enough that an unqualified point estimate would overstate the evidence.
    """
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    if len(correct) == 0:
        return float("nan"), float("nan")
    boots = [correct[rng.integers(0, len(correct), len(correct))].mean()
             for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
