# When Does Context Pay Off?

**TF-IDF, Word2Vec and BERT for Sentiment Classification under Label, Compute and Domain Constraints**

Code and data pipeline for an NLP research poster (University of Trier, NLP module).

---

## Research question

> The ranking TF-IDF < Word2Vec < BERT is established on large, clean, in-domain benchmarks.
> **Under what conditions does that ranking actually hold, and where does it break?**

| | Hypothesis | Experiment |
|---|---|---|
| **H1** | Contextual embeddings dominate only above a label-budget threshold; below ~500 labelled reviews TF-IDF is competitive or better | `exp1_label_efficiency` |
| **H2** | Accuracy gain per unit of compute is strongly diminishing; TF-IDF is on the Pareto frontier and BERT is not | `exp2_cost` |
| **H3** | BERT's advantage is concentrated in negation, contrast and out-of-domain text — the headline number *understates* where context helps | `exp3_robustness` |

H1 and H3 can come out false. That is the point: a flipped or null result is reportable.

## Experimental design

**Independent variable — the representation.** Five conditions, collapsed to three
families on the poster:

| ID | Representation | Dim |
|---|---|---|
| `tfidf` | TF-IDF, 1–2 grams, sublinear tf, 50k features | ~50k sparse |
| `w2v-self` | Word2Vec skip-gram trained on IMDB itself, mean-pooled | 300 |
| `w2v-pre` | GloVe 300d pretrained, mean-pooled | 300 |
| `bert-cls` | DistilBERT frozen, `[CLS]` vector | 768 |
| `bert-mean` | DistilBERT frozen, mask-aware mean pooling | 768 |

**Held constant — the classifier.** Every condition feeds the *same*
`LogisticRegression`, with `C` selected on the same dev split. No condition gets
a different head, so accuracy differences are attributable to the representation
alone.

**Data.** IMDB (Maas et al., 2011), official 25k/25k split. 5k is carved out of
train as a dev set for selecting `C`; the test set is touched once, at final
evaluation. Two out-of-domain sets are used for H3 only and never trained on:
Rotten Tomatoes (same domain, much shorter texts — isolates a length shift) and
Twitter US Airline (full domain shift, class-balanced to keep accuracy comparable).

## An important limitation, stated up front

**BERT is not fine-tuned here.** It is used as a frozen feature extractor, because
the study is CPU-only. Two consequences, both deliberate:

1. **It is the fair comparison.** Fine-tuning would confound "better representation"
   with "more trainable parameters" — the frozen setup isolates the representation.
2. **It is a lower bound on BERT's accuracy.** Frozen DistilBERT + LR reaches roughly
   85–89% on IMDB where a fine-tuned BERT reaches ~93–94%. Frozen `[CLS]` in particular
   is a known-weak sentence vector (Reimers & Gurevych, 2019), which is why both
   pooling strategies are carried.

For H2 this cuts in BERT's favour and the conclusion still holds: fine-tuning would
move the BERT point further right on the cost axis, never left. Any H1/H3 result
should be read as "frozen contextual features", not "BERT at its best".

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 1. Verify the wiring end-to-end in ~1 minute (200 docs per corpus)
python -m src.prepare --smoke

# 2. Encode every corpus with every representation. THE LONG STEP:
#    the DistilBERT pass over ~55k documents takes ~30-60 min on CPU, once.
python -m src.prepare

# 3. Experiments (minutes each - they read the cached features)
python -m src.exp1_label_efficiency
python -m src.exp2_cost
python -m src.exp3_robustness

# 4. Figures -> figures/*.pdf and *.svg
python -m src.figures
```

### Building the poster and appendix

```bash
# Set the repo URL once: writes poster/qr.svg and fills the URL into both documents
python poster/make_qr.py --url https://github.com/YOU/YOUR-REPO

python poster/measure.py          # reports per-column overflow in mm (A1 is 594mm tall)
python poster/build_poster.py     # poster.html -> poster/poster.pdf, exact DIN A1
python poster/build_poster.py --html poster/appendix.html --out poster/appendix.pdf
python poster/check_poster.py     # verifies the hard requirements; non-zero on failure

python poster/make_submission.py --student-id 255678   # -> 255678.zip
```

`check_poster.py` enforces the requirements the guidelines grade 5.0 for missing:
exact A1 page size, a single page, body text ≥ 24pt, all three mandatory sections
present, a real repository URL, and no leftover `TODO` placeholders.
`make_submission.py` refuses to build the ZIP while any of those fail.

Results land in `results/` as CSVs. **Every number on the poster traces to a row in
one of those files.** Seeds are fixed (`config.SEED = 42`) and recorded per row.

## Layout

```
src/config.py     paths, seeds, the representation grid, the poster palette
src/data.py       corpus loading, the fixed split protocol, H3 slice definitions
src/represent.py  the five encoders behind one interface, with disk caching
src/evaluate.py   the shared LR head, C selection, bootstrap CIs
src/prepare.py    one-off: fit + encode + cache everything
src/exp{1,2,3}_*  the three experiments
src/figures.py    the three poster figures as vector PDFs
results/          committed CSVs - the evidence behind every claim
figures/          committed vector PDFs used on the poster
artifacts/        cached embeddings (gitignored, regenerable via src.prepare)
```

## Sanity checks

Bugs in this kind of pipeline show up as *suspiciously good* numbers, so the
expected ranges are worth stating:

- TF-IDF + LR on full IMDB: **~88–90%**
- Frozen mean-pooled DistilBERT + LR: **~85–89%**
- Majority-class baseline: **50%** (the data is balanced)

Anything far above those ranges indicates label leakage or a split error, not a finding.

## References

Maas et al. (2011) *Learning Word Vectors for Sentiment Analysis* — IMDB.
Mikolov et al. (2013) *Efficient Estimation of Word Representations in Vector Space* — word2vec.
Pennington et al. (2014) *GloVe: Global Vectors for Word Representation*.
Devlin et al. (2019) *BERT: Pre-training of Deep Bidirectional Transformers*.
Sanh et al. (2019) *DistilBERT, a distilled version of BERT*.
Reimers & Gurevych (2019) *Sentence-BERT* — frozen `[CLS]` as a weak sentence vector.

Full reference list with the ethics and cost literature is in the poster appendix.
