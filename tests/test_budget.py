import numpy as np

from irteval.budget import panel, panel_orders, per_bench, plug_in
from irteval.datasets import folds, model_groups


def test_panels_are_nested_and_stratified():
    bench = np.array(["a"] * 100 + ["b"] * 300)
    ub = np.unique(bench)
    orders = panel_orders(bench, ub, seed=3)
    small, large = panel(orders, 0.05, len(bench)), panel(orders, 0.2, len(bench))
    assert np.all(large[small])
    assert small[bench == "a"].sum() == 5 and small[bench == "b"].sum() == 15


def test_plug_in_uses_observed_responses_and_predictions():
    Y = np.array([[1.0, 0.0, 1.0, 0.0]])
    W = np.ones_like(Y, bool)
    S = np.array([[True, True, False, False]])
    P = np.full_like(Y, 0.25)
    bench = np.array(["x"] * 4)
    est = plug_in(Y, W, S, P, bench, np.unique(bench))
    assert est[0, 0] == (1 + 0 + 0.25 + 0.25) / 4


def test_per_bench_ignores_unobserved_entries():
    Y = np.array([[1.0, 1.0, 0.0, 0.0]])
    mask = np.array([[True, False, True, True]])
    bench = np.array(["x", "x", "y", "y"])
    assert per_bench(Y, mask, bench, np.unique(bench)).tolist() == [[1.0, 0.0]]


def test_lineage_groups_and_grouped_folds():
    rng = np.random.default_rng(0)
    Y = (rng.random((8, 200)) < 0.5).astype(float)
    Y[1] = Y[0]                                  # near-identical response pattern
    W = np.ones_like(Y, bool)
    names = np.array(["orgA__m1", "orgB__m2", "orgC__m3", "orgC__m4", "orgD__m5", "orgE__m6", "orgF__m7", "orgG__m8"])
    g = model_groups(Y, W, names)
    assert g[0] == g[1] and g[2] == g[3] and len(np.unique(g)) == 6
    f = folds(g, k=3, grouped=True)
    assert f[0] == f[1] and f[2] == f[3]


def test_ridge_leave_one_out_error_matches_refits():
    from irteval.budget import Ridge

    rng = np.random.default_rng(13)
    X, T = rng.normal(size=(8, 3)), rng.uniform(size=(8, 1))

    def explicit_loo(lam):
        errs = []
        for i in range(len(X)):
            keep = np.arange(len(X)) != i
            xm, tm = X[keep].mean(0), T[keep].mean(0)
            Xc = X[keep] - xm
            alpha = np.linalg.solve(Xc @ Xc.T + lam * np.eye(keep.sum()), T[keep] - tm)
            errs.append(float(((T[i] - (X[i] - xm) @ Xc.T @ alpha - tm) ** 2).sum()))
        return np.mean(errs)

    best = Ridge.GRID[np.argmin([explicit_loo(lam) for lam in Ridge.GRID])]
    model = Ridge().fit(X, T)
    Xc = X - X.mean(0)
    alpha = np.linalg.solve(Xc @ Xc.T + best * np.eye(len(X)), T - T.mean(0))
    assert np.allclose(model.alpha, alpha)
