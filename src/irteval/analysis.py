"""Aggregate RQ1-RQ4 results into CSV tables and figures."""
import glob
import json
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .paths import FIGURES as FIG
from .paths import RESULTS as RES
from .paths import TABLES as TAB
from .paths import ensure_dirs

B_BOOT = 2000
CONFIRM_BUDGETS = [0.01, 0.05, 0.10]
CONFIRM_METHODS = ["DAA", "R1", "P2", "MD", "LS", "KN"]


def cluster_boot(values_by_model, groups, B=B_BOOT, seed=0):
    """Bootstrap the mean of per-model values over lineage groups."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    sums = np.array([values_by_model[groups == g].sum() for g in ug])
    cnts = np.array([(groups == g).sum() for g in ug])
    idx = rng.integers(0, len(ug), size=(B, len(ug)))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def ratio_boot(num, den, groups, B=B_BOOT, seed=0):
    """Cluster bootstrap of sum(num)/sum(den) over lineage groups."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    sn = np.array([num[groups == g].sum() for g in ug])
    sd = np.array([den[groups == g].sum() for g in ug])
    idx = rng.integers(0, len(ug), size=(B, len(ug)))
    return sn[idx].sum(1) / sd[idx].sum(1)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0
    for i, j in enumerate(order):
        run = max(run, (m - i) * pvals[j])
        adj[j] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ RQ2

def rq2_tables():
    rows = []
    for ds in ("d1", "d2"):
        f = RES / f"rq2_{ds}.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        groups = np.array(r["groups"])
        for pattern in ("mcar", "block"):
            runs = [x for x in r["runs"] if x["pattern"] == pattern]
            if not runs:
                continue
            methods = list(runs[0]["methods"])
            agg = {m: {k: sum(np.array(x["methods"][m][k]) for x in runs) for k in ("ll", "brier", "acc", "n")}
                   for m in methods}
            n = agg[methods[0]]["n"].astype(float)
            ref = agg["R1"]
            for m in methods:
                a = agg[m]
                ll = a["ll"].sum() / n.sum()
                bootll = ratio_boot(a["ll"], n, groups)
                # cluster bootstrap of the paired difference vs R1 (ratio of sums)
                rng = np.random.default_rng(1)
                ug = np.unique(groups)
                gs = {k: np.array([v[groups == g].sum() for g in ug]) for k, v in
                      (("d", a["ll"] - ref["ll"]), ("db", a["brier"] - ref["brier"]), ("n", n))}
                idx = rng.integers(0, len(ug), size=(B_BOOT, len(ug)))
                dll = gs["d"][idx].sum(1) / gs["n"][idx].sum(1)
                dbr = gs["db"][idx].sum(1) / gs["n"][idx].sum(1)
                rows.append({
                    "dataset": ds, "pattern": pattern, "method": m,
                    "logloss": ll, "logloss_lo": np.quantile(bootll, .025), "logloss_hi": np.quantile(bootll, .975),
                    "brier": a["brier"].sum() / n.sum(), "acc": a["acc"].sum() / n.sum(),
                    "auc": float(np.mean([x["methods"][m]["auc"] for x in runs])),
                    "d_logloss_vs_R1": (a["ll"] - ref["ll"]).sum() / n.sum(),
                    "d_logloss_lo": np.quantile(dll, .025), "d_logloss_hi": np.quantile(dll, .975),
                    "d_brier_vs_R1": (a["brier"] - ref["brier"]).sum() / n.sum(),
                    "d_brier_lo": np.quantile(dbr, .025), "d_brier_hi": np.quantile(dbr, .975),
                    "fit_seconds": float(np.mean([x["methods"][m]["seconds"] for x in runs])),
                })
    write_csv(rows, TAB / "rq2.csv")
    return rows


# ------------------------------------------------------------------ RQ3/RQ4

def load_budget(ds, scheme):
    files = sorted(glob.glob(str(RES / f"budget_{ds}_{scheme}_fold*.npy")))
    if not files:
        return None
    recs = [np.load(f, allow_pickle=True).item() for f in files]
    return recs


def budget_analysis(ds, scheme):
    recs = load_budget(ds, scheme)
    if recs is None:
        return None
    base = recs[0]
    truth = base["truth"]                  # models x benchmarks
    Wc = base["W_counts"]
    groups = base["groups"]
    ub = list(base["bench"])
    budgets, reps = base["budgets"], base["reps"]
    n_models = truth.shape[0]
    complete = sorted(set(np.concatenate([r["test_idx"] for r in recs]))) == list(range(n_models))

    def micro(e, wc):
        return (e * wc).sum(1) / wc.sum(1)

    def macro(e):
        return e.mean(1)

    tq = [j for j, b in enumerate(ub) if b.startswith("truthfulqa")]
    keep = [j for j in range(len(ub)) if j not in tq]
    true_micro = micro(truth, Wc)
    true_macro = macro(truth)
    true_notq = micro(truth[:, keep], Wc[:, keep])
    methods = list(recs[0]["est"][f"{budgets[0]}|0"]["acc"].keys())

    est = {}   # (f, r, method) -> dict of arrays over all test models (NaN if fold missing)
    for f in budgets:
        for r in range(reps):
            for m in methods:
                mi = np.full(n_models, np.nan)
                ma = np.full(n_models, np.nan)
                nt = np.full(n_models, np.nan)
                for rec in recs:
                    e = rec["est"][f"{f}|{r}"]["acc"][m].astype(float)
                    t = rec["test_idx"]
                    mi[t] = micro(e, Wc[t])
                    ma[t] = macro(e)
                    nt[t] = micro(e[:, keep], Wc[t][:, keep])
                est[(f, r, m)] = (mi, ma, nt)

    rows = []
    perm = {}
    for f in budgets:
        for m in methods:
            maes, maes_ma, maes_nt, taus, rhos, within = [], [], [], [], [], []
            err_by_model = np.full(n_models, np.nan)    # NaN for models outside the evaluated folds
            for r in range(reps):
                mi, ma, nt = est[(f, r, m)]
                ok = ~np.isnan(mi)
                err = np.abs(mi - true_micro)
                err_by_model[ok] = np.nan_to_num(err_by_model[ok]) + err[ok] / reps
                maes.append(100 * err[ok].mean())
                maes_ma.append(100 * np.abs(ma - true_macro)[ok].mean())
                maes_nt.append(100 * np.abs(nt - true_notq)[ok].mean())
                within.append((err[ok] <= 0.01).mean())
                taus.append(kendalltau(mi[ok], true_micro[ok], variant="b")[0])
                rhos.append(spearmanr(mi[ok], true_micro[ok])[0])
            perm[(f, m)] = err_by_model
            signed = np.mean([est[(f, r, m)][0] - true_micro for r in range(reps)], axis=0)
            ok_all = ~np.isnan(signed)
            n_items = np.mean([np.mean(rec["est"][f"{f}|0"]["n_items"]) for rec in recs])
            rows.append({"dataset": ds, "scheme": scheme, "budget": f, "method": m, "items": n_items,
                         "mae": np.mean(maes), "mae_lo": np.quantile(maes, .025), "mae_hi": np.quantile(maes, .975),
                         "mae_macro": np.mean(maes_ma), "mae_noTQA": np.mean(maes_nt),
                         "within1pp": np.mean(within), "tau": np.mean(taus),
                         "tau_lo": np.quantile(taus, .025), "tau_hi": np.quantile(taus, .975),
                         "spearman": np.mean(rhos),
                         "bias_pp": 100 * np.nanmean(signed),
                         "r_signed_vs_acc": float(np.corrcoef(signed[ok_all], true_micro[ok_all])[0, 1])})

    # paired cluster-bootstrap differences vs RAW
    diffs = []
    for f in budgets:
        for m in methods:
            if m == "RAW":
                continue
            d = perm[(f, m)] - perm[(f, "RAW")]
            evaluated = ~np.isnan(d)
            d = d[evaluated]
            bs = cluster_boot(d, groups[evaluated], seed=int(f * 1e4))
            p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            diffs.append({"dataset": ds, "scheme": scheme, "budget": f, "method": m,
                          "d_mae_pp": 100 * d.mean(), "lo": 100 * np.quantile(bs, .025),
                          "hi": 100 * np.quantile(bs, .975), "p_boot": max(p, 1 / B_BOOT)})
    fam = [x for x in diffs if x["budget"] in CONFIRM_BUDGETS and x["method"] in CONFIRM_METHODS]
    if fam:
        adj = holm(np.array([x["p_boot"] for x in fam]))
        for x, a in zip(fam, adj):
            x["p_holm"] = a

    # ability stability (within fold, same calibration)
    stab = []
    for f in budgets:
        for name in ("R1", "P2", "P2-ADA"):
            rho, rmse = [], []
            for rec in recs:
                ref_name = "P2" if name == "P2-ADA" else name
                full = rec["theta_full"][ref_name]
                sd = rec["theta_train_sd"][ref_name]
                for r in range(reps):
                    th = rec["est"][f"{f}|{r}"]["theta"][name]
                    rho.append(spearmanr(th, full)[0])
                    rmse.append(np.sqrt(((th - full) ** 2).mean()) / sd)
            stab.append({"dataset": ds, "scheme": scheme, "budget": f, "method": name,
                         "spearman_theta": np.mean(rho), "rmse_theta_sd": np.mean(rmse)})

    cost = defaultdict(list)
    for rec in recs:
        for k, v in rec["cost"].items():
            cost[k].append(v)
    lam = {str(f): {m: [rec["lambda"][f][m] for rec in recs] for m in recs[0]["lambda"][f]} for f in budgets}
    return {"rows": rows, "diffs": diffs, "stab": stab, "complete": complete, "cost": dict(cost),
            "lambda": lam, "n_folds": len(recs)}


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r.get(k, "")) for k in keys) + "\n")


def _fmt(v):
    if isinstance(v, float) or isinstance(v, np.floating):
        return f"{v:.6g}"
    return str(v)


# ------------------------------------------------------------------ figures

COLORS = {"RAW": "#222222", "DAA": "#8c6d31", "RIDGE": "#7f7f7f", "RAW-EB": "#bcbd22",
          "R1": "#1f77b4", "P2": "#2ca02c", "P3f": "#17becf", "MD": "#d62728", "LS": "#9467bd",
          "KN": "#ff7f0e", "B-MD": "#e377c2", "B-P2": "#98df8a", "P2-ADA": "#2ca02c",
          "B-LS": "#c5b0d5", "B-KN": "#ffbb78"}
STYLE = {"P2-ADA": "--", "B-MD": ":", "B-P2": ":", "B-LS": ":", "B-KN": ":", "RAW-EB": "-."}


def fig_budget(results, metric, ylabel, fname, methods, logy=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.3), sharey=False)
    for ax, ds, title in zip(axes, ("d1", "d2"), ("(a) D1: Open LLM Leaderboard", "(b) D2: EmbedLLM")):
        res = results.get((ds, "grouped"))
        if res is None:
            continue
        for m in methods:
            rr = sorted([x for x in res["rows"] if x["method"] == m], key=lambda x: x["budget"])
            if not rr:
                continue
            x = [100 * q["budget"] for q in rr]
            ax.plot(x, [q[metric] for q in rr], STYLE.get(m, "-"), marker="o", ms=2.5, lw=1.1,
                    color=COLORS.get(m, None), label={"R1": "Rasch", "P2": "2PL", "MD": "MIRT"}.get(m, m))
            if metric + "_lo" in rr[0]:
                ax.fill_between(x, [q[metric + "_lo"] for q in rr], [q[metric + "_hi"] for q in rr],
                                color=COLORS.get(m, None), alpha=0.12, lw=0)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 20, 40, 80])
        ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "40", "80"])
        ax.set_xlabel("Evaluation budget, % of items")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, lw=0.5)
    axes[1].legend(fontsize=9, ncol=2, frameon=False, handlelength=1.6, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=300)
    plt.close(fig)


NAMES = {"arc": "ARC-C", "gsm8k": "GSM8K", "hellaswag": "HellaSwag", "mmlu": "MMLU", "truthfulqa": "TruthfulQA",
         "winogrande": "WinoGrande", "asdiv": "ASDiv", "gpqa": "GPQA", "logiqa": "LogiQA", "mathqa": "MathQA",
         "medmcqa": "MedMCQA", "piqa": "PIQA", "social_iqa": "SocialIQA", "truthfulqa_mc1": "TruthfulQA-MC1"}


def fig_loadings():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.6), gridspec_kw={"height_ratios": [6, 10]})
    for ax, ds, title in zip(axes, ("d1", "d2"), ("(a) D1, d = 8", "(b) D2, d = 5")):
        f = RES / f"rq1_{ds}.json"
        if not f.exists():
            continue
        L = json.load(open(f))["mirt_loadings"]
        M = np.array(L["mean_loading"])
        v = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(L["bench"])))
        ax.set_yticklabels([NAMES.get(b, b) for b in L["bench"]], fontsize=10)
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels([f"F{j + 1}" for j in range(M.shape[1])])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}".replace("-0.00", "0.00"), ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j]) > 0.6 * v else "black")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "fig_mirt_loadings.png", dpi=300)
    plt.close(fig)


def main():
    ensure_dirs()
    rq2_tables()
    results = {}
    all_rows, all_diffs, all_stab = [], [], []
    summary = {}
    for ds in ("d1", "d2"):
        for scheme in ("grouped", "random"):
            res = budget_analysis(ds, scheme)
            if res is None:
                continue
            results[(ds, scheme)] = res
            all_rows += res["rows"]
            all_diffs += res["diffs"]
            all_stab += res["stab"]
            summary[f"{ds}_{scheme}"] = {"complete": res["complete"], "n_folds": res["n_folds"],
                                        "cost": res["cost"], "lambda": res["lambda"]}
    write_csv(all_rows, TAB / "budget.csv")
    write_csv(all_diffs, TAB / "budget_diffs.csv")
    write_csv(all_stab, TAB / "stability.csv")
    json.dump(summary, open(TAB / "budget_summary.json", "w"), indent=1, default=float)
    main_methods = ["RAW", "DAA", "R1", "P2", "MD", "LS", "KN"]
    if results:
        fig_budget(results, "mae", "MAE of accuracy, p.p.", "fig_budget_mae.png", main_methods, logy=True)
        fig_budget(results, "tau", "Kendall tau-b", "fig_budget_tau.png", main_methods)
    fig_loadings()

