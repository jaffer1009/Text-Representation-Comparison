"""Experiment 1 (H1): how much labelled data does each representation need?

H1: contextual embeddings dominate only above a label budget threshold; below
~500 labelled reviews, sparse TF-IDF is competitive or better.

Design: 8 label budgets x 5 representations x 5 seeds = 200 fits, all on cached
features. Output: results/exp1_label_efficiency.csv (one row per fit).

    python -m src.exp1_label_efficiency [--smoke]
"""
from __future__ import annotations

import argparse
import itertools

import pandas as pd

from .config import RESULTS, SEEDS, TRAIN_SIZES, REPRESENTATIONS
from .data import load_all, subsample
from . import represent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", nargs="*", default=list(REPRESENTATIONS))
    args = ap.parse_args()

    from .evaluate import fit_and_score

    fit_c, dev_c, test_c, _ = load_all(smoke=args.smoke, with_ood=False)
    sizes = [50, 100] if args.smoke else TRAIN_SIZES
    seeds = SEEDS[:2] if args.smoke else SEEDS

    rows = []
    for rep_id in args.reps:
        rep = represent.build(rep_id)
        # Features are already cached by src.prepare; transform() is a disk read.
        Xfit = rep.transform(fit_c)
        Xdev = rep.transform(dev_c)
        Xtest = rep.transform(test_c)

        for n, seed in itertools.product(sizes, seeds):
            if n > len(fit_c):
                continue
            idx = subsample(n, fit_c.y, seed)
            metrics, best_C, _ = fit_and_score(
                Xfit[idx], fit_c.y[idx], Xdev, dev_c.y,
                [("imdb-test", Xtest, test_c.y)], seed=seed,
            )
            m = metrics["imdb-test"]
            rows.append({
                "representation": rep_id,
                "family": REPRESENTATIONS[rep_id]["family"],
                "n_train": n,
                "seed": seed,
                "C": best_C,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
            })
            print(f"  {rep_id:10} n={n:<6} seed={seed}  acc={m['accuracy']:.4f}")

    df = pd.DataFrame(rows)
    out = RESULTS / ("exp1_label_efficiency_smoke.csv" if args.smoke
                     else "exp1_label_efficiency.csv")
    df.to_csv(out, index=False)
    print(f"\n[exp1] wrote {out}  ({len(df)} rows)")

    if not df.empty:
        print("\nMean accuracy by representation x n_train:")
        print(df.pivot_table(index="n_train", columns="representation",
                             values="accuracy", aggfunc="mean").round(4))


if __name__ == "__main__":
    main()
