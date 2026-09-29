"""Experiment 2 (H2): what does each representation cost?

H2: accuracy gain per unit of compute is strongly diminishing - TF-IDF sits on
the accuracy/latency Pareto frontier and BERT does not.

Three costs are separated, because they are paid at different times:

  build     one-off: fitting the vectorizer / training word2vec / loading weights
  head fit  per training run, on cached features
  latency   per prediction, raw text -> label, INCLUDING encoding

Latency is the one that matters for deployment and is what Figure 2 plots. It is
measured end to end with the cache deliberately bypassed, because a real system
does not have its future inputs pre-encoded.

The BERT numbers are a LOWER BOUND on its true cost: the model is frozen here, so
fine-tuning would add training time and move the BERT point further right on the
cost axis, never left.

    python -m src.exp2_cost [--n-latency 1000] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd

from .config import ARTIFACTS, REPRESENTATIONS, RESULTS
from .data import Corpus, load_all
from . import represent


def _median_of(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-latency", type=int, default=500,
                    help="documents per end-to-end latency measurement")
    ap.add_argument("--reps", nargs="*", default=list(REPRESENTATIONS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from .evaluate import fit_head

    fit_c, dev_c, test_c, _ = load_all(smoke=args.smoke, with_ood=False)
    n_lat = min(args.n_latency, len(test_c))
    probe = Corpus("latency-probe", test_c.texts[:n_lat], test_c.y[:n_lat])

    rows = []
    for rep_id in args.reps:
        rep = represent.build(rep_id)

        # Build cost, measured here rather than read from prepare.py, so this
        # experiment is self-contained and the encoder is fitted for latency.
        t0 = time.perf_counter()
        rep.fit(fit_c)
        build_s = time.perf_counter() - t0

        Xfit, Xdev = rep.transform(fit_c), rep.transform(dev_c)
        head_s = _median_of(lambda: fit_head(Xfit, fit_c.y, Xdev, dev_c.y), repeats=1)
        pipe, best_C = fit_head(Xfit, fit_c.y, Xdev, dev_c.y)

        # End-to-end: encode raw text, then predict. Cache bypassed on purpose.
        latency_s = _median_of(lambda: pipe.predict(rep._transform(probe)), repeats=3)

        cache_mb = sum(p.stat().st_size for p in ARTIFACTS.glob(f"{rep_id}__*")) / 1e6
        rows.append({
            "representation": rep_id,
            "family": REPRESENTATIONS[rep_id]["family"],
            "build_seconds": round(build_s, 2),
            "head_fit_seconds": round(head_s, 2),
            "latency_seconds_per_doc": latency_s / n_lat,
            f"latency_seconds_per_{n_lat}_docs": round(latency_s, 3),
            "feature_dim": int(Xfit.shape[1]),
            "cached_features_mb": round(cache_mb, 1),
            "C": best_C,
        })
        print(f"  {rep_id:10} build={build_s:7.1f}s  "
              f"latency={latency_s:6.2f}s/{n_lat}docs  dim={Xfit.shape[1]}")

    df = pd.DataFrame(rows)
    out = RESULTS / ("exp2_cost_smoke.csv" if args.smoke else "exp2_cost.csv")
    df.to_csv(out, index=False)
    (RESULTS / "exp2_machine.json").write_text(json.dumps({
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "n_latency_docs": n_lat,
        "note": "BERT figures are frozen-feature costs; fine-tuning would add training time.",
    }, indent=2))
    print(f"\n[exp2] wrote {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
