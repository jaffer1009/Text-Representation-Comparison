"""Experiment 3 (H3): where does contextualisation actually help?

H3: BERT's advantage is concentrated in negation, contrast and out-of-domain
text - i.e. the headline i.i.d. number understates where context matters.

Each representation is trained ONCE on the full IMDB pool, then evaluated on:
  - the full IMDB test set (the reference number)
  - linguistic slices of that test set (negation / contrast / length)
  - two out-of-domain corpora, zero-shot

Reported as a delta against the representation's own overall IMDB accuracy, so
the question is "where does this model lose ground relative to itself?" rather
than "which model is best", which Experiment 1 already answers.

    python -m src.exp3_robustness
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import REPRESENTATIONS, RESULTS, SEED
from .data import load_all, slice_masks
from . import represent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=list(REPRESENTATIONS))
    ap.add_argument("--n-examples", type=int, default=5,
                    help="qualitative examples to dump for the poster's error box")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from .evaluate import bootstrap_ci, fit_and_score

    fit_c, dev_c, test_c, ood = load_all(smoke=args.smoke)
    masks = slice_masks(test_c)

    rows, pred_store = [], {}
    for rep_id in args.reps:
        rep = represent.build(rep_id)
        Xfit, Xdev, Xtest = rep.transform(fit_c), rep.transform(dev_c), rep.transform(test_c)

        eval_sets = [("imdb-test", Xtest, test_c.y)]
        for c in ood:
            eval_sets.append((c.name, rep.transform(c), c.y))

        metrics, best_C, preds = fit_and_score(
            Xfit, fit_c.y, Xdev, dev_c.y, eval_sets, seed=SEED
        )
        pred_store[rep_id] = preds["imdb-test"]
        overall = metrics["imdb-test"]["accuracy"]

        def record(slice_name, y_true, y_pred, kind):
            acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean()) if len(y_true) else np.nan
            lo, hi = bootstrap_ci(y_true, y_pred)
            rows.append({
                "representation": rep_id,
                "family": REPRESENTATIONS[rep_id]["family"],
                "slice": slice_name,
                "kind": kind,
                "n": int(len(y_true)),
                "accuracy": acc,
                "ci_lo": lo,
                "ci_hi": hi,
                "delta_vs_overall": acc - overall,
                "C": best_C,
            })

        record("imdb-test (all)", test_c.y, preds["imdb-test"], "reference")
        for name, mask in masks.items():
            record(name, test_c.y[mask], preds["imdb-test"][mask], "linguistic")
        for c in ood:
            record(c.name, c.y, preds[c.name], "out-of-domain")

        print(f"  {rep_id:10} imdb={overall:.4f}  " +
              "  ".join(f"{c.name}={metrics[c.name]['accuracy']:.3f}" for c in ood))

    df = pd.DataFrame(rows)
    out = RESULTS / ("exp3_robustness_smoke.csv" if args.smoke else "exp3_robustness.csv")
    df.to_csv(out, index=False)
    print(f"\n[exp3] wrote {out}")

    # Qualitative error box: reviews where BERT is right and TF-IDF is wrong,
    # and the reverse. These go on the poster verbatim (truncated).
    if "bert-mean" in pred_store and "tfidf" in pred_store:
        b, t, y = pred_store["bert-mean"], pred_store["tfidf"], test_c.y
        neg_mask = masks["negation"]
        picks = []
        for label, sel in [("bert_right_tfidf_wrong", (b == y) & (t != y) & neg_mask),
                           ("tfidf_right_bert_wrong", (t == y) & (b != y) & neg_mask)]:
            for i in np.flatnonzero(sel)[: args.n_examples]:
                picks.append({
                    "case": label,
                    "gold": int(y[i]),
                    "text": test_c.texts[i][:400].replace("\n", " "),
                })
        pd.DataFrame(picks).to_csv(RESULTS / "exp3_examples.csv", index=False)
        print(f"[exp3] wrote {RESULTS / 'exp3_examples.csv'}")


if __name__ == "__main__":
    main()
