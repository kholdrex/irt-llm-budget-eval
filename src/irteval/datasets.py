"""Response matrices, lineage groups, cross-validation folds and the IRT model factory."""
import json
from pathlib import Path

import numpy as np

from . import models as mdl
from .paths import DATA

# Fixed lower asymptote 1/k of the 3PL-c variant, fitted on D1 only (multiple-choice benchmarks
# scored by option log-likelihood); zero for thresholded TruthfulQA-mc2 and GSM8K.
GUESS_D1 = {"mmlu": 0.25, "arc": 0.25, "hellaswag": 0.25, "winogrande": 0.5,
            "truthfulqa": 0.0, "gsm8k": 0.0}

DATASETS = {"d1": "openllm", "d2": "embedllm"}


def load(ds):
    """Return (Y, W, bench, models, guess): 0/1 responses, observation mask, benchmark of each item,
    model identifiers and the fixed lower asymptote per item (D1 only)."""
    z = np.load(DATA / f"{DATASETS[ds]}.npz")
    Yraw = z["Y"]
    W = Yraw >= 0
    Y = np.where(W, Yraw, 0).astype(np.float64)
    bench = z["bench"]
    models = z["models"]
    guess = np.array([GUESS_D1[b] for b in bench]) if ds == "d1" else None
    return Y, W, bench, models, guess


def model_groups(Y, W, models, thr=0.98):
    """Connected components of: same uploader OR response agreement >= thr."""
    n = len(models)
    org = np.array([m.split("__")[0] for m in models])
    Yf, Wf = Y.astype(np.float32), W.astype(np.float32)
    agree = (Yf @ Yf.T + (Wf - Yf) @ (Wf - Yf).T) / np.maximum(Wf @ Wf.T, 1)
    adj = (agree >= thr) | (org[:, None] == org[None, :])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(adj, 1))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    _, g = np.unique(roots, return_inverse=True)
    return g


def folds(groups, k=5, seed=0, grouped=True):
    """Assign models to k folds; grouped=True keeps each group in one fold
    (greedy balancing of fold sizes over randomly ordered groups)."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    if not grouped:
        f = np.empty(n, int)
        f[rng.permutation(n)] = np.arange(n) % k
        return f
    ug = rng.permutation(np.unique(groups))
    sizes = np.zeros(k)
    gf = {}
    for g in sorted(ug, key=lambda g: -(groups == g).sum()):
        j = int(np.argmin(sizes))
        gf[g] = j
        sizes[j] += (groups == g).sum()
    return np.array([gf[g] for g in groups])


def make_irt(name, guess=None, d=None):
    """R1 = Rasch, P2 = 2PL, P3f = 2PL with a fixed lower asymptote, MD = MIRT with dimension d."""
    if name == "R1":
        return mdl.IRT("rasch")
    if name == "P2":
        return mdl.IRT("2pl")
    if name == "P3f":
        return mdl.IRT("2pl", guess=guess)
    if name == "MD":
        return mdl.IRT("mirt", d=d)
    raise ValueError(name)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
