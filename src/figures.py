"""All three poster figures, as vector PDFs.

Font sizes are set for A1: these figures are placed at roughly 25-30 cm wide, so
what looks oversized on screen reads correctly on the printed poster. Vector
output means the 150 PPI requirement is satisfied by construction.

Colour is carried by REP_STYLE in config.py - the same colour means the same
representation in every figure and in the poster body text.

    python -m src.figures
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FAMILY_COLORS, FIGURES, REPRESENTATIONS, REP_STYLE, RESULTS

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,          # embed real text, not outlines
})

GRID = dict(alpha=0.25, linewidth=0.6)


def _save(fig, name: str, suffix: str) -> None:
    """Write both PDF and SVG.

    The SVG is what the poster HTML embeds - it stays vector through Chrome's
    print-to-PDF, so the >=150 PPI requirement is met by construction rather than
    by choosing a big enough raster size.
    """
    for ext in ("pdf", "svg"):
        fig.savefig(FIGURES / f"{name}{suffix}.{ext}")
    plt.close(fig)
    print(f"[figures] {name}{suffix}.pdf + .svg")


def fig1_learning_curves(suffix: str = "") -> None:
    path = RESULTS / f"exp1_label_efficiency{suffix}.csv"
    if not path.exists():
        print("[figures] skipping fig1 (run exp1 first)")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for rep_id, g in df.groupby("representation"):
        s = g.groupby("n_train")["accuracy"]
        mean, sem = s.mean(), s.std(ddof=1) / np.sqrt(s.count())
        style = REP_STYLE[rep_id]
        ax.plot(mean.index, mean.values, marker="o", markersize=4,
                color=style["color"], linestyle=style["ls"],
                label=REPRESENTATIONS[rep_id]["label"], linewidth=2)
        ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=style["color"], alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Labelled training reviews (log scale)")
    ax.set_ylabel("Accuracy on IMDB test (25k)")
    ax.set_title("H1  Label efficiency: when does context start paying off?")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.text(df.n_train.min(), 0.505, "majority-class baseline", fontsize=10, color="grey")
    ax.grid(**GRID)
    ax.legend(frameon=False, loc="lower right")

    _annotate_crossover(ax, df)
    _save(fig, "fig1_label_efficiency", suffix)


def _annotate_crossover(ax, df: pd.DataFrame) -> None:
    """Mark where the TF-IDF and best-BERT curves actually cross.

    Both directions matter and the interesting one is whichever happens LAST:
    on this data BERT leads at the smallest budgets and TF-IDF overtakes it
    later, so annotating the first crossing would report the opposite story to
    the one the results actually tell.
    """
    piv = df.pivot_table(index="n_train", columns="representation",
                         values="accuracy", aggfunc="mean")
    bert_cols = [c for c in piv.columns if c.startswith("bert")]
    if "tfidf" not in piv.columns or not bert_cols:
        return

    bert = piv[bert_cols].max(axis=1)
    tfidf = piv["tfidf"]
    tfidf_ahead = (tfidf > bert).values
    sizes = piv.index.values

    # Every index where the lead changes hands.
    flips = [i for i in range(1, len(sizes)) if tfidf_ahead[i] != tfidf_ahead[i - 1]]
    if not flips:
        leader = "TF-IDF" if tfidf_ahead[-1] else "frozen BERT"
        ax.text(0.02, 0.95, f"{leader} leads at every budget tested",
                transform=ax.transAxes, fontsize=12, va="top",
                bbox=dict(boxstyle="round", fc="#fff3cd", ec="#d9a406"))
        return

    i = flips[-1]                      # the decisive, last crossing
    n = int(sizes[i])
    gained = "TF-IDF overtakes BERT" if tfidf_ahead[i] else "BERT overtakes TF-IDF"
    ax.axvline(n, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.annotate(f"{gained}\nat n ≈ {n:,}",
                xy=(n, tfidf.iloc[i]), xytext=(14, -50),
                textcoords="offset points", fontsize=11,
                arrowprops=dict(arrowstyle="->", alpha=0.6))


def fig2_cost(suffix: str = "") -> None:
    cost_p = RESULTS / f"exp2_cost{suffix}.csv"
    acc_p = RESULTS / f"exp1_label_efficiency{suffix}.csv"
    if not (cost_p.exists() and acc_p.exists()):
        print("[figures] skipping fig2 (run exp1 and exp2 first)")
        return
    cost = pd.read_csv(cost_p)
    acc = (pd.read_csv(acc_p).query("n_train == n_train.max()")
           .groupby("representation")["accuracy"].mean())

    lat_col = [c for c in cost.columns if c.startswith("latency_seconds_per_")][0]
    n_docs = lat_col.split("_")[-2]
    cost = cost[cost.representation.isin(acc.index)]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for _, r in cost.iterrows():
        rep = r["representation"]
        ax.scatter(r[lat_col], acc[rep], s=190, zorder=3,
                   color=REP_STYLE[rep]["color"],
                   edgecolor="white", linewidth=1.5)
        ax.annotate(REPRESENTATIONS[rep]["label"],
                    (r[lat_col], acc[rep]), xytext=(9, 6),
                    textcoords="offset points", fontsize=11)

    # Pareto frontier: points not dominated on both cheaper and more accurate.
    pts = sorted(((r[lat_col], acc[r["representation"]]) for _, r in cost.iterrows()))
    frontier, best = [], -np.inf
    for x, y in pts:
        if y > best:
            frontier.append((x, y))
            best = y
    if len(frontier) > 1:
        ax.plot(*zip(*frontier), color="grey", linestyle="--",
                linewidth=1.2, zorder=1, label="Pareto frontier")
        ax.legend(frameon=False, loc="lower right")

    ax.set_xscale("log")
    ax.set_xlabel(f"Inference latency, seconds per {n_docs} documents (log scale)")
    ax.set_ylabel("Accuracy at full training budget")
    ax.set_title("H2  Cost: is the accuracy gain worth the compute?")
    ax.grid(**GRID)
    _save(fig, "fig2_cost", suffix)


def fig3_robustness(suffix: str = "") -> None:
    path = RESULTS / f"exp3_robustness{suffix}.csv"
    if not path.exists():
        print("[figures] skipping fig3 (run exp3 first)")
        return
    df = pd.read_csv(path)
    df = df[df["kind"] != "reference"]

    order = [s for s in ["negation", "contrast", "short (<128w)", "long (>400w)",
                         "rotten-tomatoes", "twitter-airline"]
             if s in set(df["slice"])]
    reps = [r for r in REPRESENTATIONS if r in set(df["representation"])]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.8 / len(reps)
    x = np.arange(len(order))
    for i, rep in enumerate(reps):
        sub = df[df.representation == rep].set_index("slice").reindex(order)
        ax.bar(x + i * width - 0.4 + width / 2, sub["delta_vs_overall"],
               width * 0.92, color=REP_STYLE[rep]["color"],
               label=REPRESENTATIONS[rep]["label"],
               hatch="//" if REP_STYLE[rep]["ls"] == "--" else None,
               edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Accuracy change vs. own IMDB score")
    ax.set_title("H3  Robustness: where does each representation lose ground?")
    ax.grid(axis="y", **GRID)
    ax.legend(frameon=False, ncol=2, loc="lower left")

    if len(order) > 4:  # separate the linguistic slices from the domain shift
        ax.axvline(3.5, color="grey", linestyle=":", linewidth=1)
        ax.text(3.6, ax.get_ylim()[1] * 0.92, "out-of-domain",
                fontsize=11, color="grey")

    _save(fig, "fig3_robustness", suffix)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="read *_smoke.csv and write *_smoke.pdf, leaving real figures alone")
    sfx = "_smoke" if ap.parse_args().smoke else ""

    fig1_learning_curves(sfx)
    fig2_cost(sfx)
    fig3_robustness(sfx)
    print("->", FIGURES)
