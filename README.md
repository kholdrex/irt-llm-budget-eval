# irt-llm-budget-eval

Code for the study *Comparing IRT models and simple baselines for budget-constrained LLM evaluation*.

The study compares item response models with simple non-psychometric estimators on public model × item
correctness matrices of large language models, asking how well each approach estimates item difficulty,
held-out responses, the capability of a new model and its full-benchmark accuracy and rank when only
0.5–80% of the benchmark items are evaluated.

## Methods

| Code | Method |
|---|---|
| `R1` | Rasch (1PL) |
| `P2` | two-parameter model (2PL), positive discriminations |
| `P3f` | 2PL with a fixed lower asymptote `1/k` (multiple-choice benchmarks of D1) |
| `MD` | multidimensional IRT (MIRT) |
| `LS` | low-rank least-squares completion with item and model intercepts |
| `KN` | k nearest models |
| `IM`, `MM` | item mean, model mean |
| `RAW` | stratified subset accuracy |
| `DAA` | difficulty-adjusted accuracy (difference estimator) |
| `RIDGE`, `RAW-EB` | ridge regression from subset responses; empirical-Bayes shrinkage of `RAW` |
| `B-*` | blend `λ·RAW + (1 − λ)·X`, with `λ` tuned on an inner grouped split |
| `P2-ADA` | two-stage adaptive item selection by 2PL Fisher information |

IRT models are fitted by penalised joint maximum likelihood (Gaussian priors, L-BFGS-B with analytic
gradients). Abilities of new models are estimated with item parameters fixed (Fisher scoring with an
empirical-Bayes prior). Model-based estimators of accuracy keep the observed responses and replace the
unobserved ones by predicted probabilities.

## Data

| ID | Source | Licence | Size |
|---|---|---|---|
| D1 | Open LLM Leaderboard v1 correctness released with [tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks) (`tutorials/data/lb.pickle`) | MIT (repository) | 395 models × 28,659 items |
| D2 | [EmbedLLM](https://huggingface.co/datasets/RZ412/EmbedLLM) correctness data (`train/val/test.csv`) | Apache-2.0 | 112 models × 35,673 items |

Both sources are downloaded at pinned revisions and checked (SHA-256 for D1, row counts for D2). Only the
0/1 correctness of responses is used. TruthfulQA in D1 is scored as `mc2 > 0.5`. In D2, prompts listed
under several task variants are resolved by majority label; ties are treated as missing.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Python ≥ 3.10. Dependencies are pinned in `pyproject.toml`.

## Reproducing the results

All outputs are written below the working directory (current directory, or `IRTEVAL_WORKDIR`):
`data/`, `results/`, `results/tables/`, `figures/`.

```bash
irteval prepare                 # download data, build data/openllm.npz and data/embedllm.npz
irteval rq2 d1 && irteval rq2 d2                # held-out responses; selects MIRT dimension, LS rank, k
irteval rq1 d1 && irteval rq1 d2                # item difficulty analyses
irteval budget d1 && irteval budget d2          # reduced budgets, grouped 5-fold cross-validation
irteval analyze                 # CSV tables and figures
irteval report                  # paper tables (Markdown) and the signed-error figure
```

`make all` runs the same sequence. On an 8-core laptop the full pipeline takes several hours, most of it in
`budget d1`; folds can be run in separate processes, e.g. `irteval budget d1 --folds 0 1`. The secondary
interpolation setting is available as `irteval budget d1 --scheme random`.

| Paper element | Output |
|---|---|
| Table 2 (held-out responses) | `results/tables/rq2.csv` |
| Tables 3–4, Fig. 1 (accuracy and ranking under budgets) | `results/tables/budget.csv`, `figures/fig_budget_mae.png` |
| Paired differences vs `RAW`, Holm-adjusted | `results/tables/budget_diffs.csv` |
| Ability stability | `results/tables/stability.csv` |
| Table 5, Fig. 3 (item difficulty, MIRT loadings) | `results/rq1_d*.json`, `figures/fig_mirt_loadings.png` |
| Fig. 2 (signed errors) | `figures/fig_bias.png` |
| All tables as Markdown | `results/tables/paper_tables.md` |

## Layout

```
src/irteval/
  prepare.py    data download and response matrices
  datasets.py   loading, lineage groups, cross-validation folds
  models.py     IRT models, low-rank completion, nearest models, metrics
  rq1.py        item difficulty experiments
  rq2.py        held-out response prediction
  budget.py     reduced-budget evaluation of new models
  analysis.py   aggregation, bootstrap intervals, figures
  report.py     paper tables
  cli.py        command-line interface
tests/          gradient checks, parameter recovery, estimator unit tests
```

## License

MIT, see `LICENSE`.
