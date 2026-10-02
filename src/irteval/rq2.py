"""RQ2: prediction of held-out model x item entries.

Patterns: "mcar"  - 10% of observed entries uniformly at random;
          "block" - for 20% of models one whole benchmark is hidden.
Hyperparameters (MIRT d, LS rank, KN k) are selected once per dataset on a 5% validation
split of the training entries of pattern mcar / seed 0 and then kept fixed.
Saves per-model sufficient statistics so that cluster-bootstrap intervals can be computed.
"""
import time

import numpy as np

from . import models as mdl
from .datasets import load, make_irt, model_groups, save_json
from .logs import get_logger
from .paths import RESULTS, ensure_dirs

D_GRID = [2, 3, 5, 8]
R_GRID = [1, 2, 3, 5, 8, 12]
K_GRID = [5, 10, 20, 40]
SEEDS = [0, 1, 2]


def split(W, bench, pattern, seed):
    rng = np.random.default_rng(1000 + seed)
    test = np.zeros_like(W)
    if pattern == "mcar":
        test = W & (rng.random(W.shape) < 0.10)
    else:
        M = W.shape[0]
        chosen = rng.choice(M, size=int(round(0.2 * M)), replace=False)
        ub = np.unique(bench)
        for m in chosen:
            b = rng.choice(ub)
            test[m] = W[m] & (bench == b)
    return W & ~test, test


def fit_predict(name, Y, Wtr, guess, hp):
    t = time.time()
    if name == "IM":
        P = np.broadcast_to(mdl.item_mean(Y, Wtr), Y.shape)
    elif name == "MM":
        P = np.broadcast_to(mdl.model_mean(Y, Wtr)[:, None], Y.shape)
    elif name == "LS":
        P = mdl.LowRankLS(hp["LS"]).fit(Y, Wtr).predict()
    elif name == "KN":
        kn = mdl.KNN(hp["KN"]).fit(Y, Wtr)
        P = kn.predict_for(Y, Wtr, exclude_self=np.arange(Y.shape[0]))
    else:
        m = make_irt(name, guess, d=hp.get("MD"))
        m.fit(Y, Wtr)
        P = m.predict()
        return np.clip(P, mdl.EPS, 1 - mdl.EPS), time.time() - t, m.n_iter
    return np.clip(P, mdl.EPS, 1 - mdl.EPS), time.time() - t, None


def per_model_stats(Y, P, test):
    y = np.where(test, Y, 0.0)
    ll = -np.where(test, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0).sum(1)
    br = np.where(test, (P - y) ** 2, 0.0).sum(1)
    acc = np.where(test, (P >= 0.5) == (y == 1), False).sum(1)
    return {"ll": ll, "brier": br, "acc": acc, "n": test.sum(1)}


def select_hp(Y, W, bench, guess, log):
    Wtr, _ = split(W, bench, "mcar", 0)
    rng = np.random.default_rng(77)
    val = Wtr & (rng.random(W.shape) < 0.05)
    Wfit = Wtr & ~val
    yv = Y[val]
    out = {"MD": {}, "LS": {}, "KN": {}}
    for d in D_GRID:
        P, sec, it = fit_predict("MD", Y, Wfit, guess, {"MD": d})
        out["MD"][d] = mdl.log_loss(yv, P[val])
        log(f"  select MD d={d}: val logloss {out['MD'][d]:.5f} ({sec:.0f}s, {it} it)")
    for r in R_GRID:
        P, sec, _ = fit_predict("LS", Y, Wfit, guess, {"LS": r})
        out["LS"][r] = mdl.brier(yv, P[val])
        log(f"  select LS r={r}: val brier {out['LS'][r]:.5f} ({sec:.0f}s)")
    for k in K_GRID:
        P, sec, _ = fit_predict("KN", Y, Wfit, guess, {"KN": k})
        out["KN"][k] = mdl.log_loss(yv, P[val])
        log(f"  select KN k={k}: val logloss {out['KN'][k]:.5f} ({sec:.0f}s)")
    hp = {"MD": min(out["MD"], key=out["MD"].get), "LS": min(out["LS"], key=out["LS"].get),
          "KN": min(out["KN"], key=out["KN"].get)}
    return hp, out


def main(ds):
    """Hyperparameter selection and held-out prediction for data set ds ("d1" or "d2")."""
    ensure_dirs()
    Y, W, bench, models, guess = load(ds)
    log = get_logger(f"rq2_{ds}", RESULTS / f"rq2_{ds}.log")
    groups = model_groups(Y, W, models)
    log(f"{ds}: {Y.shape}, groups {groups.max() + 1}")
    hp, sel = select_hp(Y, W, bench, guess, log)
    log(f"selected {hp}")
    methods = ["IM", "MM", "R1", "P2"] + (["P3f"] if guess is not None else []) + ["MD", "LS", "KN"]
    out = {"dataset": ds, "shape": Y.shape, "groups": groups, "hp": hp, "selection": sel, "runs": []}
    for pattern in ["mcar", "block"]:
        for seed in SEEDS:
            Wtr, test = split(W, bench, pattern, seed)
            run = {"pattern": pattern, "seed": seed, "n_test": int(test.sum()), "methods": {}}
            for name in methods:
                P, sec, it = fit_predict(name, Y, Wtr, guess, hp)
                st = per_model_stats(Y, P, test)
                st["auc"] = mdl.auc(Y[test], P[test])
                st["seconds"] = sec
                st["iters"] = it
                run["methods"][name] = st
                log(f"{pattern} s{seed} {name:4s} ll {st['ll'].sum() / st['n'].sum():.4f} "
                    f"brier {st['brier'].sum() / st['n'].sum():.4f} auc {st['auc']:.4f} ({sec:.0f}s)")
            out["runs"].append(run)
            save_json(out, RESULTS / f"rq2_{ds}.json")

