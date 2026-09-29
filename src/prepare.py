"""One-off: fit every representation on the IMDB training pool and cache the
encoded matrices for every corpus.

This is the only expensive step in the project (the DistilBERT pass over ~55k
documents on CPU). Everything downstream reads the cached `.npy`/`.npz` files,
which is what makes 200+ model fits across three experiments affordable.

    python -m src.prepare                 # all representations
    python -m src.prepare --only tfidf    # one
    python -m src.prepare --smoke         # tiny subset, proves the wiring works
"""
from __future__ import annotations

import argparse
import json
import time

from .config import ARTIFACTS, REPRESENTATIONS
from .data import load_all
from . import represent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(REPRESENTATIONS))
    ap.add_argument("--smoke", action="store_true",
                    help="200 docs per corpus; verifies the pipeline end-to-end in ~1 min")
    ap.add_argument("--no-ood", action="store_true")
    args = ap.parse_args()

    fit, dev, test, ood = load_all(smoke=args.smoke, with_ood=not args.no_ood)
    corpora = [fit, dev, test] + ood

    print(f"[prepare] corpora: {[(c.name, len(c)) for c in corpora]}")

    timings = {}

    # BERT's two conditions share one forward pass - by far the longest step.
    bert_ids = [r for r in args.only if r.startswith("bert")]
    if bert_ids:
        load_s = represent.encode_bert_both(fit, corpora)
        for rep_id in bert_ids:
            timings[rep_id] = {"fit_seconds": load_s}

    for rep_id in [r for r in args.only if not r.startswith("bert")]:
        t0 = time.perf_counter()
        rep = represent.build(rep_id).fit(fit)
        timings[rep_id] = {"fit_seconds": time.perf_counter() - t0}
        print(f"[prepare] {rep_id}: fitted in {timings[rep_id]['fit_seconds']:.1f}s")
        for c in corpora:
            rep.transform(c)

    (ARTIFACTS / "fit_timings.json").write_text(json.dumps(timings, indent=2))
    print("[prepare] done ->", ARTIFACTS)


if __name__ == "__main__":
    main()
