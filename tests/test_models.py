import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import expit

import irteval.models as mdl


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(1)
    theta = rng.standard_normal((60, 3))
    a = np.abs(rng.standard_normal((300, 3))) * 0.8
    b = rng.standard_normal(300)
    Y = (rng.random((60, 300)) < expit(theta @ a.T - b)).astype(float)
    W = rng.random((60, 300)) < 0.9
    return Y, W


def objective(monkeypatch, model, Y, W):
    """Capture the objective and starting point that IRT.fit passes to the optimizer."""
    captured = {}

    def fake_minimize(f, x0, **kwargs):
        captured.update(f=f, x0=x0)

        class Result:
            x, nit = x0, 0
        return Result()

    monkeypatch.setattr(mdl, "minimize", fake_minimize)
    model.fit(Y, W)
    return captured["f"], captured["x0"]


@pytest.mark.parametrize("kind,d", [("rasch", 1), ("2pl", 1), ("mirt", 3)])
@pytest.mark.parametrize("guess", [False, True])
def test_analytic_gradient(monkeypatch, synthetic, kind, d, guess):
    Y, W = synthetic[0][:10, :40], synthetic[1][:10, :40]
    c = np.full(40, 0.25) if guess else None
    f, x0 = objective(monkeypatch, mdl.IRT(kind, d, guess=c, dtype=np.float64, positive=False), Y, W)
    x = x0 + 0.1 * np.random.default_rng(0).standard_normal(x0.size)
    err = check_grad(lambda v: f(v)[0], lambda v: f(v)[1], x)
    assert err / np.linalg.norm(f(x)[1]) < 1e-4


def test_rasch_parameter_recovery():
    rng = np.random.default_rng(1)
    theta, b = rng.standard_normal(200), rng.standard_normal(500)
    Y = (rng.random((200, 500)) < expit(theta[:, None] - b[None])).astype(float)
    m = mdl.IRT("rasch").fit(Y, np.ones_like(Y, bool))
    assert np.corrcoef(m.theta, theta)[0, 1] > 0.98
    assert np.corrcoef(m.b, b)[0, 1] > 0.98


@pytest.mark.parametrize("kind,d", [("rasch", 1), ("2pl", 1), ("mirt", 3)])
def test_fold_in_reproduces_joint_abilities(synthetic, kind, d):
    Y, W = synthetic
    m = mdl.IRT(kind, d).fit(Y, W)
    theta = m.fold_in(Y, W, prior="standard")
    assert np.corrcoef(np.ravel(theta), np.ravel(m.theta))[0, 1] > 0.99


def test_predictions_are_probabilities(synthetic):
    Y, W = synthetic
    for model in (mdl.IRT("2pl", guess=np.full(Y.shape[1], 0.25)).fit(Y, W), mdl.LowRankLS(3).fit(Y, W)):
        P = model.predict()
        assert P.shape == Y.shape and np.all((P > 0) & (P < 1))


def test_lowrank_fold_in_matches_fit(synthetic):
    Y, W = synthetic
    ls = mdl.LowRankLS(3).fit(Y, W)
    u, c = ls.fold_in(Y[:5], W[:5])
    assert np.abs(ls.predict(u, c) - ls.predict()[:5]).mean() < 0.01


def test_knn_excludes_self(synthetic):
    Y, W = synthetic
    kn = mdl.KNN(1).fit(Y, W)
    P = kn.predict_for(Y, W, exclude_self=np.arange(len(Y)))
    assert P.shape == Y.shape and np.all((P > 0) & (P < 1))


def test_metrics():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    assert mdl.auc(y, p) == pytest.approx(0.75)
    assert mdl.brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    assert mdl.log_loss(np.array([1.0]), np.array([0.5])) == pytest.approx(np.log(2))
