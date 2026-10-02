"""RQ3/RQ4: estimating a new model's capability and full-benchmark accuracy from a
fraction of the items.

Outer 5-fold cross-validation over models ("grouped": lineage groups kept together;
"random": interpolation setting). In each fold, item parameters are calibrated on the
training models using all items. Repetition r draws one shared item panel per budget f
(benchmark-stratified, proportional allocation, nested across budgets); a test model
observes the panel items it has responses for. Per-benchmark accuracy estimates:

  RAW     panel mean
  RAW-EB  panel mean shrunk to the training-model mean (normal-normal empirical Bayes)
  DAA     difficulty-adjusted accuracy: mean(y_S) - mean(p_S) + mean(p_U), p = item rate
  RIDGE   ridge regression from panel responses to benchmark accuracy (training models)
  R1, P2, P3f, MD, LS, KN   plug-in: (sum_S y + sum_{U \\ S} p_hat) / |U|
  P2-ADA  two-stage adaptive panel per model: half random, half max 2PL information
  B-<X>   lambda * RAW + (1 - lambda) * X, lambda per budget from inner grouped validation

Micro accuracy = item-weighted average over benchmarks; macro = unweighted average.
"""
import json
import time

import numpy as np

from . import models as mdl
from .datasets import folds, load, make_irt, model_groups
from .logs import get_logger
from .paths import RESULTS, ensure_dirs

BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
REPS = 20
INNER_REPS = 5
LAMBDAS = np.round(np.linspace(0, 1, 21), 2)
BLEND = ["P2", "MD", "LS", "KN"]


def panel_orders(bench, ub, seed):
    rng = np.random.default_rng(seed)
    return {b: rng.permutation(np.flatnonzero(bench == b)) for b in ub}


def panel(orders, f, Q):
    P = np.zeros(Q, bool)
    for idx in orders.values():
        P[idx[:max(1, int(round(f * len(idx))))]] = True
    return P


def per_bench(values, mask, bench, ub):
    """Mean of `values` over `mask` within each benchmark -> (models x benchmarks)."""
    out = np.zeros((values.shape[0], len(ub)))
    for j, b in enumerate(ub):
        c = bench == b
        out[:, j] = (values[:, c] * mask[:, c]).sum(1) / np.maximum(mask[:, c].sum(1), 1)
    return out


def plug_in(Y, W, S, P, bench, ub):
    return per_bench(np.where(S, Y, P), W, bench, ub)


class Ridge:
    """Dual ridge regression, penalty chosen per output by exact leave-one-out error."""
    GRID = 10.0 ** np.arange(-2, 5)

    def fit(self, X, T):
        self.xm, self.tm = X.mean(0), T.mean(0)
        Xc, Tc = X - self.xm, T - self.tm
        K = Xc @ Xc.T
        evals, evecs = np.linalg.eigh(K)
        best = np.full(T.shape[1], np.inf)
        self.alpha = np.zeros_like(Tc)
        for lam in self.GRID:
            Hd = evecs @ np.diag(evals / (evals + lam)) @ evecs.T
            A = evecs @ np.diag(1 / (evals + lam)) @ evecs.T @ Tc
            # leverage of the centred fit plus 1/n for the intercept
            loo = ((Tc - Hd @ Tc) / (1 - 1 / len(X) - np.diag(Hd))[:, None]) ** 2
            err = loo.mean(0)
            upd = err < best
            best[upd] = err[upd]
            self.alpha[:, upd] = A[:, upd]
        self.Xc = Xc
        return self

    def predict(self, X):
        return np.clip((X - self.xm) @ self.Xc.T @ self.alpha + self.tm, 0, 1)


def adaptive_mask(irt, Y, W, P_half, n_total):
    """Second stage: add the most informative items (2PL Fisher information at the interim
    ability) until each model has n_total observed items."""
    S = W & P_half[None, :]
    th = irt.fold_in(Y, S)
    info = irt.item_information(th)
    info[~W | S] = -np.inf
    for i in range(W.shape[0]):
        need = n_total[i] - S[i].sum()
        if need > 0:
            S[i, np.argsort(-info[i])[:need]] = True
    return S


class Fitted:
    def __init__(self, Ytr, Wtr, guess, hp, names, log, tag):
        self.irt, self.cost = {}, {}
        for name in names:
            m = make_irt(name, guess, d=hp["MD"]).fit(Ytr, Wtr)
            self.irt[name] = m
            self.cost[name] = {"seconds": m.seconds, "iters": m.n_iter,
                               "params": m.n_params(*Ytr.shape)}
            log(f"  {tag} {name} fitted in {m.seconds:.0f}s ({m.n_iter} it)")
        t = time.time()
        self.ls = mdl.LowRankLS(hp["LS"]).fit(Ytr, Wtr)
        self.cost["LS"] = {"seconds": time.time() - t}
        self.kn = mdl.KNN(hp["KN"]).fit(Ytr, Wtr)
        self.p_item = mdl.item_mean(Ytr, Wtr)
        self.Ytr, self.Wtr = Ytr, Wtr


def estimates(F, Y, W, Pf, bench, ub, with_baselines=True):
    S = W & Pf[None, :]
    out, thetas = {}, {}
    out["RAW"] = per_bench(Y, S, bench, ub)
    for name, m in F.irt.items():
        th = m.fold_in(Y, S)
        if name in ("R1", "P2"):
            thetas[name] = th
        out[name] = plug_in(Y, W, S, m.predict(th), bench, ub)
    u, c = F.ls.fold_in(Y, S)
    out["LS"] = plug_in(Y, W, S, F.ls.predict(u, c), bench, ub)
    out["KN"] = plug_in(Y, W, S, F.kn.predict_for(Y, S), bench, ub)
    if with_baselines:
        tr_acc = per_bench(F.Ytr, F.Wtr, bench, ub)
        n = np.maximum(np.array([S[:, bench == b].sum(1) for b in ub]).T, 1)
        tau2 = tr_acc.var(0)[None, :]
        r = (out["RAW"] * n + 0.5) / (n + 1)
        w = tau2 / (tau2 + r * (1 - r) / n)
        out["RAW-EB"] = w * out["RAW"] + (1 - w) * tr_acc.mean(0)[None, :]
        pS = per_bench(np.broadcast_to(F.p_item, Y.shape), S, bench, ub)
        pU = per_bench(np.broadcast_to(F.p_item, Y.shape), W, bench, ub)
        out["DAA"] = np.clip(out["RAW"] - pS + pU, 0, 1)
        Xtr = np.where(F.Wtr[:, Pf], F.Ytr[:, Pf], F.p_item[Pf][None, :])
        Xte = np.where(S[:, Pf], Y[:, Pf], F.p_item[Pf][None, :])
        out["RIDGE"] = Ridge().fit(Xtr, tr_acc).predict(Xte)
    return out, thetas, S


def micro(est, W, bench, ub):
    n = np.array([W[:, bench == b].sum(1) for b in ub]).T
    return (est * n).sum(1) / n.sum(1)


def run_fold(scheme, k, Y, W, bench, guess, groups, fold_of, hp, log):
    ub = np.unique(bench)
    Q = Y.shape[1]
    tr, te = fold_of != k, fold_of == k
    names = ["R1", "P2"] + (["P3f"] if guess is not None else []) + ["MD"]
    F = Fitted(Y[tr], W[tr], guess, hp, names, log, f"fold {k}")

    # inner grouped validation for blend weights
    tr_idx = np.flatnonzero(tr)
    inner = folds(groups[tr_idx], 5, seed=100 + k, grouped=True)
    itr, iva = tr_idx[inner != 0], tr_idx[inner == 0]
    Fi = Fitted(Y[itr], W[itr], guess, hp, ["P2", "MD"], log, f"fold {k} inner")
    lam = {}
    truth_iva = micro(per_bench(Y[iva], W[iva], bench, ub), W[iva], bench, ub)
    for f in BUDGETS:
        errs = {m: np.zeros(len(LAMBDAS)) for m in BLEND}
        for r in range(INNER_REPS):
            Pf = panel(panel_orders(bench, ub, [k, r, 999]), f, Q)
            est, _, _ = estimates(Fi, Y[iva], W[iva], Pf, bench, ub, with_baselines=False)
            raw = micro(est["RAW"], W[iva], bench, ub)
            for m in BLEND:
                x = micro(est[m], W[iva], bench, ub)
                for j, l in enumerate(LAMBDAS):
                    errs[m][j] += np.abs(l * raw + (1 - l) * x - truth_iva).mean()
        lam[f] = {m: float(LAMBDAS[np.argmin(errs[m])]) for m in BLEND}
    log(f"  fold {k} blend lambdas {lam}")

    Yte, Wte = Y[te], W[te]
    theta_full = {n: F.irt[n].fold_in(Yte, Wte) for n in ("R1", "P2")}
    rec = {"test_idx": np.flatnonzero(te), "theta_full": theta_full,
           "theta_train_sd": {n: float(F.irt[n].theta.std()) for n in ("R1", "P2")},
           "est": {}, "cost": F.cost, "lambda": lam, "bench": ub}
    for f in BUDGETS:
        for r in range(REPS):
            orders = panel_orders(bench, ub, [r, 12345])
            Pf = panel(orders, f, Q)
            est, thetas, S = estimates(F, Yte, Wte, Pf, bench, ub)
            for m in BLEND:
                est["B-" + m] = lam[f][m] * est["RAW"] + (1 - lam[f][m]) * est[m]
            Ph = panel(orders, f / 2, Q)
            n_total = S.sum(1)
            Sa = adaptive_mask(F.irt["P2"], Yte, Wte, Ph, n_total)
            tha = F.irt["P2"].fold_in(Yte, Sa)
            thetas["P2-ADA"] = tha
            est["P2-ADA"] = plug_in(Yte, Wte, Sa, F.irt["P2"].predict(tha), bench, ub)
            rec["est"][f"{f}|{r}"] = {"acc": {m: v.astype(np.float32) for m, v in est.items()},
                                      "theta": thetas, "n_items": n_total}
        log(f"  fold {k} budget {f} done")
    return rec


def main(ds, scheme="grouped", which=None):
    """Run the budget experiment for data set ds ("d1" or "d2") and fold scheme "grouped" or "random"."""
    ensure_dirs()
    which = list(range(5)) if not which else which
    Y, W, bench, models, guess = load(ds)
    hp = {k: int(v) for k, v in json.load(open(RESULTS / f"rq2_{ds}.json"))["hp"].items()}
    groups = model_groups(Y, W, models)
    fold_of = folds(groups, 5, seed=0, grouped=(scheme == "grouped"))
    log = get_logger(f"budget_{ds}_{scheme}", RESULTS / f"budget_{ds}_{scheme}.log")
    ub = np.unique(bench)
    log(f"{ds} {scheme}: hp {hp}, fold sizes {np.bincount(fold_of)}")
    truth = per_bench(Y, W, bench, ub)
    for k in which:
        rec = run_fold(scheme, k, Y, W, bench, guess, groups, fold_of, hp, log)
        rec.update({"truth": truth, "W_counts": np.array([W[:, bench == b].sum(1) for b in ub]).T,
                    "groups": groups, "fold_of": fold_of, "budgets": BUDGETS, "reps": REPS,
                    "hp": hp})
        np.save(RESULTS / f"budget_{ds}_{scheme}_fold{k}.npy", rec, allow_pickle=True)
