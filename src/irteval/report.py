"""Paper tables (Markdown) and the signed-error figure, generated from the aggregated results.

Writes results/tables/paper_tables.md and figures/fig_bias.png; run after `irteval analyze`.
"""
import csv
import json

import numpy as np

from . import analysis as analyze
from .paths import FIGURES
from .paths import RESULTS as RES
from .paths import TABLES as TAB

LABEL = {"RAW": "RAW", "DAA": "DAA", "R1": "Rasch", "P2": "2PL", "P3f": "3PL-c", "MD": "MIRT",
         "LS": "LS", "KN": "KN", "IM": "IM", "MM": "MM", "RIDGE": "RIDGE", "B-MD": "MIRT+RAW blend",
         "P2-ADA": "2PL-ADA", "RAW-EB": "RAW-EB"}
MAIN = ["RAW", "DAA", "R1", "P2", "MD", "LS", "KN"]
BUDGETS = [0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 0.80]


def read(name):
    with open(TAB / name) as f:
        return list(csv.DictReader(f))


def fmt(x, d=2):
    return f"{float(x):.{d}f}"


def table_rq2():
    rows = read("rq2.csv")
    get = {(r["dataset"], r["pattern"], r["method"]): r for r in rows}
    methods = ["IM", "MM", "R1", "P2", "P3f", "MD", "LS", "KN"]
    out = ["Table 2 – Prediction of held-out responses: log loss / Brier score (mean over three splits)", "",
           "| Method | D1, hidden cells | D1, hidden benchmark | D2, hidden cells | D2, hidden benchmark |",
           "|---|---|---|---|---|"]
    for m in methods:
        cells = []
        for ds in ("d1", "d2"):
            for pat in ("mcar", "block"):
                r = get.get((ds, pat, m))
                cells.append("–" if r is None else f"{fmt(r['logloss'], 3)} / {fmt(r['brier'], 3)}")
        out.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_budget(metric, title, digits, budgets=None):
    rows = [r for r in read("budget.csv") if r["scheme"] == "grouped"]
    get = {(r["dataset"], float(r["budget"]), r["method"]): r for r in rows}
    cols = MAIN + ["RIDGE", "B-MD"]
    out = [title, "", "| Budget (items) | " + " | ".join(LABEL[m] for m in cols) + " |",
           "|---|" + "---|" * len(cols)]
    for ds in ("d1", "d2"):
        out.append(f"| **{ds.upper()}** |" + " |" * len(cols))
        for f in (budgets or BUDGETS):
            n = float(get[(ds, f, "RAW")]["items"])
            vals = [get[(ds, f, m)][metric] for m in cols]
            best = min(float(v) for v in vals) if metric == "mae" else max(float(v) for v in vals)
            tol = 10 ** -(digits + 1)
            cells = [f"**{fmt(v, digits)}**" if abs(float(v) - best) < tol else fmt(v, digits) for v in vals]
            out.append(f"| {100 * f:g}% ({n:.0f}) | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_rq1():
    out = ["Table 5 – MAE of predicted item correctness rates", "",
           "| Data set, direction | 2PL | Rasch | Logit shift | Additive shift | Source rate | Noise floor |",
           "|---|---|---|---|---|---|---|"]
    r = {ds: json.load(open(RES / f"rq1_{ds}.json")) for ds in ("d1", "d2")}
    for ds in ("d1", "d2"):
        for d, lab in (("weak_to_strong", "weaker → stronger"), ("strong_to_weak", "stronger → weaker"),
                       ("random_halves", "random halves")):
            res = r[ds]["1b"][d]
            v = [np.mean([x[k]["mae"] for x in res]) for k in ("P2", "R1", "LOGIT", "SHIFT", "SRC")]
            fl = np.mean([x["noise_floor_mae"] for x in res])
            best = min(v)
            cells = [f"**{x:.3f}**" if abs(x - best) < 5e-4 else f"{x:.3f}" for x in v] + [f"{fl:.3f}"]
            out.append(f"| {ds.upper()}, {lab} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def fig_bias(f=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    methods = ["R1", "P2", "MD", "LS"]
    fig, axes = plt.subplots(2, 4, figsize=(6.0, 4.4), sharex="row", sharey="row")
    for i, ds in enumerate(("d1", "d2")):
        recs = analyze.load_budget(ds, "grouped")
        base = recs[0]
        Wc = base["W_counts"]
        true = (base["truth"] * Wc).sum(1) / Wc.sum(1)
        for j, m in enumerate(methods):
            err = np.full(len(true), np.nan)
            for rec in recs:
                t = rec["test_idx"]
                e = np.mean([(rec["est"][f"{f}|{r}"]["acc"][m] * Wc[t]).sum(1) / Wc[t].sum(1)
                             for r in range(base["reps"])], axis=0)
                err[t] = e - true[t]
            ax = axes[i, j]
            ok = ~np.isnan(err)
            ax.scatter(100 * true[ok], 100 * err[ok], s=4, alpha=0.6, color=analyze.COLORS[m], lw=0)
            ax.axhline(0, color="k", lw=0.6)
            rr = np.corrcoef(true[ok], err[ok])[0, 1]
            ax.set_title(f"{'D1' if ds == 'd1' else 'D2'}: {LABEL[m]}\n(r = {rr:.2f})", fontsize=10)
            if j == 0:
                ax.set_ylabel("Signed error, p.p.")
            if i == 1:
                ax.set_xlabel("True accuracy, %")
            ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_bias.png", dpi=300)
    plt.close(fig)


def main():
    mae_title = "Table 3 – MAE of estimated micro accuracy, p.p. (grouped cross-validation, mean over 20 panels)"
    tau_title = ("Table 4 – Kendall's τ-b between estimated and true accuracy "
                 "(grouped cross-validation, mean over 20 panels)")
    parts = [table_rq2(),
             table_budget("mae", mae_title, 2, [0.005, 0.01, 0.02, 0.05, 0.10, 0.40, 0.80]),
             table_budget("tau", tau_title, 3, [0.005, 0.01, 0.05, 0.10, 0.40]),
             table_rq1()]
    (TAB / "paper_tables.md").write_text("\n\n".join(parts) + "\n")
    fig_bias()
    print((TAB / "paper_tables.md").read_text())

