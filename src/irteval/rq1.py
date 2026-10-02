"""RQ1: item difficulty — agreement with empirical difficulty, extrapolation across model
populations, slope diagnostics, MIRT loadings, and a parameter-recovery check.
"""
import json

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import spearmanr

from . import models as mdl
from .datasets import load, make_irt, model_groups, save_json
from .logs import get_logger
from .paths import RESULTS, ensure_dirs

N_ANCHOR_DRAWS = 20


def varimax(A, iters=100, tol=1e-8):
    p, k = A.shape
    R = np.eye(k)
    d = 0
    for _ in range(iters):
        L = A @ R
        u, s, vt = np.linalg.svd(A.T @ (L ** 3 - L @ np.diag((L ** 2).sum(0)) / p))
        R = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return A @ R, R


def group_halves(groups, acc, rng=None):
    """Split models into two halves keeping groups intact: by group mean accuracy, or
    randomly if rng is given."""
    ug = np.unique(groups)
    if rng is None:
        order = sorted(ug, key=lambda g: acc[groups == g].mean())
    else:
        order = list(rng.permutation(ug))
    first = np.zeros(len(groups), bool)
    for g in order:
        if first.sum() >= len(groups) / 2:
            break
        first[groups == g] = True
    return first


def stratified_items(bench, frac, rng):
    sel = np.zeros(len(bench), bool)
    for b in np.unique(bench):
        idx = np.flatnonzero(bench == b)
        sel[rng.choice(idx, max(1, int(round(frac * len(idx)))), replace=False)] = True
    return sel


def rates(Y, W):
    return (Y * W).sum(0) / np.maximum(W.sum(0), 1)


def metrics(pred, obs, n_obs):
    ok = n_obs > 0
    pred, obs = pred[ok], obs[ok]
    slope = np.polyfit(pred, obs, 1)[0] if pred.std() > 0 else np.nan
    return {"mae": float(np.abs(pred - obs).mean()), "rmse": float(np.sqrt(((pred - obs) ** 2).mean())),
            "spearman": float(spearmanr(pred, obs)[0]), "slope": float(slope)}


def logit_shift(logits, weights, target):
    """Constant shift delta such that the weighted mean of sigmoid(logits + delta) equals target."""
    def gap(delta):
        return (expit(logits + delta) * weights).sum() / weights.sum() - target
    return brentq(gap, -15, 15)


def transfer(Y, W, bench, src, tgt, guess, rng_seed):
    fits = {n: make_irt(n, guess).fit(Y[src], W[src]) for n in ("R1", "P2")}
    p_src = rates(Y[src], W[src])
    out = []
    for a in range(N_ANCHOR_DRAWS):
        rng = np.random.default_rng([rng_seed, a])
        anc = stratified_items(bench, 0.2, rng)
        rest = ~anc
        Yt, Wt = Y[tgt], W[tgt]
        Wanc = Wt & anc[None, :]
        obs = rates(Yt[:, rest], Wt[:, rest])
        n_obs = Wt[:, rest].sum(0)
        res = {}
        for n, m in fits.items():
            th = m.fold_in(Yt, Wanc)
            P = m.predict(th)[:, rest]
            res[n] = metrics((P * Wt[:, rest]).sum(0) / np.maximum(n_obs, 1), obs, n_obs)
        res["SRC"] = metrics(p_src[rest], obs, n_obs)
        gap = (Yt * Wanc).sum() / Wanc.sum() - (p_src[anc] * Wanc.sum(0)[anc]).sum() / Wanc.sum()
        res["SHIFT"] = metrics(np.clip(p_src[rest] + gap, 0, 1), obs, n_obs)
        lp = logit(np.clip(p_src, 0.005, 0.995))
        delta = logit_shift(lp[anc], Wanc.sum(0)[anc], (Yt * Wanc).sum() / Wanc.sum())
        res["LOGIT"] = metrics(expit(lp[rest] + delta), obs, n_obs)
        p = np.clip(obs, 0, 1)
        floor = np.sqrt(2 / np.pi) * np.sqrt(p * (1 - p) / np.maximum(n_obs, 1))
        res["noise_floor_mae"] = float(floor[n_obs > 0].mean())
        out.append(res)
    return out


def main(ds):
    """Difficulty analyses for data set ds ("d1" or "d2"); requires the RQ2 results (selected MIRT dimension)."""
    ensure_dirs()
    log = get_logger(f"rq1_{ds}", RESULTS / f"rq1_{ds}.log")
    Y, W, bench, models, guess = load(ds)
    hp = json.load(open(RESULTS / f"rq2_{ds}.json"))["hp"]
    groups = model_groups(Y, W, models)
    acc = (Y * W).sum(1) / W.sum(1)
    ub = np.unique(bench)
    out = {"dataset": ds}
    p_item = rates(Y, W)

    # 1a: agreement with empirical difficulty (descriptive)
    r1 = make_irt("R1", guess).fit(Y, W)
    p2 = make_irt("P2", guess).fit(Y, W)
    out["1a"] = {"R1": float(spearmanr(r1.b, 1 - p_item)[0]),
                 "P2": float(spearmanr(p2.b, 1 - p_item)[0]),
                 "P2_by_bench": {b: float(spearmanr(p2.b[bench == b], 1 - p_item[bench == b])[0]) for b in ub},
                 "R1_b_sd": float(r1.b.std()), "P2_a_median": float(np.median(p2.a)),
                 "P2_a_at_bound_share": float((p2.a <= 1.01e-3).mean())}
    log(f"1a: Spearman R1 {out['1a']['R1']:.4f}, P2 {out['1a']['P2']:.4f}")

    # 1c: signed-slope 2PL diagnostics
    p2s = mdl.IRT("2pl", positive=False).fit(Y, W)
    out["1c"] = {b: {"n": int((bench == b).sum()),
                     "a_median": float(np.median(p2s.a[bench == b])),
                     "a_q10": float(np.quantile(p2s.a[bench == b], 0.1)),
                     "a_q90": float(np.quantile(p2s.a[bench == b], 0.9)),
                     "share_a_le_0": float((p2s.a[bench == b] <= 0).mean()),
                     "share_a_lt_0.2": float((p2s.a[bench == b] < 0.2).mean())} for b in ub}
    out["1c_all"] = {"share_a_le_0": float((p2s.a <= 0).mean()), "n_items": int(len(p2s.a))}
    log(f"1c: {out['1c_all']}")

    # MIRT loadings (varimax) by benchmark
    md = make_irt("MD", guess, d=int(hp["MD"])).fit(Y, W)
    L, _ = varimax(md.a)
    L = L * np.sign(L.sum(0))[None, :]
    out["mirt_loadings"] = {"d": int(hp["MD"]), "bench": list(ub),
                            "mean_loading": [[float(L[bench == b, j].mean()) for j in range(L.shape[1])] for b in ub]}

    # 1b: extrapolation stress test
    weak = group_halves(groups, acc)
    rand = group_halves(groups, acc, rng=np.random.default_rng(5))
    out["1b"] = {"halves": {"weak_n": int(weak.sum()), "weak_mean_acc": float(acc[weak].mean()),
                            "strong_mean_acc": float(acc[~weak].mean())}}
    for name, src, tgt, seed in [("weak_to_strong", weak, ~weak, 1), ("strong_to_weak", ~weak, weak, 2),
                                 ("random_halves", rand, ~rand, 3)]:
        res = transfer(Y, W, bench, src, tgt, guess, seed)
        out["1b"][name] = res
        mae = {k: round(float(np.mean([r[k]["mae"] for r in res])), 4) for k in ("R1", "P2", "SRC", "SHIFT", "LOGIT")}
        log(f"1b {name}: MAE {mae}")

    # parameter recovery of P2 at the observed size and missingness
    rng = np.random.default_rng(11)
    Ysim = (rng.random(Y.shape) < p2.predict()).astype(float)
    rec = make_irt("P2", guess).fit(Ysim, W)
    out["recovery_P2"] = {"r_theta": float(np.corrcoef(rec.theta, p2.theta)[0, 1]),
                          "r_b": float(np.corrcoef(rec.b, p2.b)[0, 1]),
                          "r_a": float(np.corrcoef(rec.a, p2.a)[0, 1]),
                          "rmse_theta": float(np.sqrt(((rec.theta - p2.theta) ** 2).mean())),
                          "rmse_b": float(np.sqrt(((rec.b - p2.b) ** 2).mean())),
                          "rmse_a": float(np.sqrt(((rec.a - p2.a) ** 2).mean()))}
    log(f"recovery: {out['recovery_P2']}")
    np.savez_compressed(RESULTS / f"rq1_{ds}_params.npz", r1_b=r1.b, r1_theta=r1.theta, p2_a=p2.a,
                        p2_b=p2.b, p2_theta=p2.theta, p2s_a=p2s.a, p_item=p_item, bench=bench,
                        mirt_L=L)
    save_json(out, RESULTS / f"rq1_{ds}.json")

