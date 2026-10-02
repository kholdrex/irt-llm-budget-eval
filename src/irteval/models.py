"""Latent-variable and baseline predictors for model x item correctness matrices.

All predictors work on a 0/1 matrix Y (models x items) with a boolean mask W of entries
that may be used for fitting. Psychometric models are fitted by penalised joint maximum
likelihood (MAP with Gaussian priors) using L-BFGS-B with analytic gradients:

    Rasch : logit p = theta_m - b_q
    2PL   : logit p = a_q (theta_m - b_q)
    MIRT  : logit p = a_q^T theta_m - b_q          (theta_m, a_q in R^d)

Priors: theta ~ N(0, I), b ~ N(0, SB^2), 2PL a ~ N(1, SA^2), MIRT a ~ N(0, I).
Optional fixed lower asymptote c_q (3PL with known guessing): p = c + (1 - c) sigmoid(.).
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

SB = 2.0
SA = 1.0
EPS = 1e-4


def _nll_terms(z, Y, W, c=None):
    """Bernoulli negative log-likelihood and its derivative w.r.t. the logit z.
    With a fixed lower asymptote c (per item): p = c + (1 - c) * sigmoid(z)."""
    s = expit(z)
    if c is None:
        ll = W * (Y * z - np.logaddexp(z.dtype.type(0.0), z))
        return -float(ll.sum(dtype=np.float64)), W * (s - Y)
    p = np.clip(c + (1 - c) * s, 1e-6, 1 - 1e-6)
    ll = W * (Y * np.log(p) + (1 - Y) * np.log1p(-p))
    dz = (1 - c) * s * (1 - s)
    return -float(ll.sum(dtype=np.float64)), W * dz * ((1 - Y) / (1 - p) - Y / p)


def _clipped_logit(x, lo=0.02, hi=0.98):
    x = np.clip(x, lo, hi)
    return np.log(x / (1 - x))


@dataclass
class IRT:
    """kind: "rasch", "2pl" or "mirt". guess: optional fixed lower asymptote per item
    (3PL with known c), used only where scoring forces a choice among k options."""
    kind: str
    d: int = 1
    guess: np.ndarray = None
    theta: np.ndarray = None
    a: np.ndarray = None
    b: np.ndarray = None
    n_iter: int = 0
    seconds: float = 0.0
    dtype: type = np.float32
    positive: bool = True        # 2PL: constrain a_q >= 1e-3 (signed fit only as a diagnostic)

    # ---------- joint calibration ----------
    def fit(self, Y, W, maxiter=1000, seed=0):
        import time
        t0_ = time.time()
        M, Q = Y.shape
        f32 = self.dtype
        Y = Y.astype(f32)
        W = W.astype(f32)
        c = None if self.guess is None else self.guess.astype(f32)[None, :]
        rng = np.random.default_rng(seed)
        pq = (W * Y).sum(0) / np.maximum(W.sum(0), 1)
        pm = (W * Y).sum(1) / np.maximum(W.sum(1), 1)
        b0 = -_clipped_logit(pq)
        t0 = _clipped_logit(pm) - _clipped_logit(pm).mean()
        if self.kind == "rasch":
            x0 = np.concatenate([t0, b0])
        elif self.kind == "2pl":
            x0 = np.concatenate([t0, np.ones(Q), b0])
        else:
            d = self.d
            th = np.column_stack([t0] + [0.1 * rng.standard_normal(M) for _ in range(d - 1)])
            A = np.column_stack([np.ones(Q)] + [0.1 * rng.standard_normal(Q) for _ in range(d - 1)])
            x0 = np.concatenate([th.ravel(), A.ravel(), b0])

        def f(x):
            th, a, b = self._unpack(x, M, Q)
            th32, b32 = th.astype(f32), b.astype(f32)
            if self.kind == "rasch":
                z = th32[:, None] - b32[None, :]
                nll, R = _nll_terms(z, Y, W, c)
                g_th = R.sum(1, dtype=np.float64) + th
                g_b = -R.sum(0, dtype=np.float64) + b / SB**2
                pen = 0.5 * (th @ th) + 0.5 * (b @ b) / SB**2
                return nll + pen, np.concatenate([g_th, g_b])
            if self.kind == "2pl":
                a32 = a.astype(f32)
                z = a32[None, :] * (th32[:, None] - b32[None, :])
                nll, R = _nll_terms(z, Y, W, c)
                Rs = R.sum(0, dtype=np.float64)
                g_th = (R @ a32).astype(np.float64) + th
                g_a = (R.T @ th32).astype(np.float64) - b * Rs + (a - 1) / SA**2
                g_b = -a * Rs + b / SB**2
                pen = 0.5 * (th @ th) + 0.5 * ((a - 1) @ (a - 1)) / SA**2 + 0.5 * (b @ b) / SB**2
                return nll + pen, np.concatenate([g_th, g_a, g_b])
            a32 = a.astype(f32)
            z = th32 @ a32.T - b32[None, :]
            nll, R = _nll_terms(z, Y, W, c)
            g_th = (R @ a32).astype(np.float64) + th
            g_a = (R.T @ th32).astype(np.float64) + a
            g_b = -R.sum(0, dtype=np.float64) + b / SB**2
            pen = 0.5 * (th * th).sum() + 0.5 * (a * a).sum() + 0.5 * (b @ b) / SB**2
            return nll + pen, np.concatenate([g_th.ravel(), g_a.ravel(), g_b])

        bounds = None
        if self.kind == "2pl" and self.positive:
            bounds = [(None, None)] * M + [(1e-3, None)] * Q + [(None, None)] * Q
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "maxcor": 20, "gtol": 1e-5, "ftol": 1e-10})
        self.n_iter = res.nit
        self.theta, self.a, self.b = self._unpack(res.x, M, Q)
        if self.kind == "2pl" and self.a.mean() < 0:      # sign convention
            self.a, self.theta, self.b = -self.a, -self.theta, -self.b
        self.seconds = time.time() - t0_
        return self

    def _unpack(self, x, M, Q):
        if self.kind == "rasch":
            return x[:M], None, x[M:]
        if self.kind == "2pl":
            return x[:M], x[M:M + Q], x[M + Q:]
        d = self.d
        th = x[:M * d].reshape(M, d)
        a = x[M * d:M * d + Q * d].reshape(Q, d)
        return th, a, x[M * d + Q * d:]

    def n_params(self, M, Q):
        return {"rasch": M + Q, "2pl": M + 2 * Q}.get(self.kind, M * self.d + Q * self.d + Q)

    # ---------- prediction ----------
    def _A(self):
        if self.kind == "rasch":
            return np.ones((len(self.b), 1))
        if self.kind == "2pl":
            return self.a[:, None]
        return self.a

    def _offset(self):
        # logit = theta^T A_q - offset_q
        return self.b if self.kind in ("rasch", "mirt") else self.a * self.b

    def predict(self, theta=None):
        th = self.theta if theta is None else theta
        th = th.reshape(len(th), -1)
        s = expit(th @ self._A().T - self._offset()[None, :])
        if self.guess is None:
            return s
        return self.guess[None, :] + (1 - self.guess[None, :]) * s

    # ---------- fold-in of new models (item parameters fixed) ----------
    def fold_in(self, Y, W, n_iter=30, prior="empirical"):
        """MAP ability of each row of Y from entries with W=1, item parameters fixed.
        Fisher scoring; prior N(mu, S) estimated from the calibration models
        ("empirical") or N(0, I) ("standard")."""
        A = self._A()
        off = self._offset()
        d = A.shape[1]
        th_cal = self.theta.reshape(len(self.theta), -1)
        if prior == "empirical":
            mu = th_cal.mean(0)
            S = np.cov(th_cal.T).reshape(d, d) + 1e-3 * np.eye(d)
        else:
            mu, S = np.zeros(d), np.eye(d)
        P = np.linalg.inv(S)
        Y = Y.astype(np.float64)
        W = W.astype(np.float64)
        c = 0.0 if self.guess is None else self.guess[None, :]
        th = np.tile(mu, (Y.shape[0], 1))
        for _ in range(n_iter):
            s = expit(th @ A.T - off[None, :])
            p = np.clip(c + (1 - c) * s, 1e-6, 1 - 1e-6)
            dp = (1 - c) * s * (1 - s)
            r = W * dp * (Y - p) / (p * (1 - p))         # score w.r.t. logit
            v = W * dp * dp / (p * (1 - p))              # Fisher information w.r.t. logit
            G = r @ A - (th - mu) @ P
            if d == 1:
                H = (v @ (A[:, 0] ** 2))[:, None, None] + P[None]
            else:
                H = np.einsum("mq,qi,qj->mij", v, A, A, optimize=True) + P[None]
            step = np.linalg.solve(H, G[..., None])[..., 0]
            th += np.clip(step, -2, 2)
            if np.abs(step).max() < 1e-7:
                break
        return th[:, 0] if self.kind in ("rasch", "2pl") else th

    def item_information(self, theta):
        """Fisher information of each item at unidimensional ability theta (rows)."""
        A = self._A()[:, 0]
        s = expit(theta[:, None] * A[None, :] - self._offset()[None, :])
        c = 0.0 if self.guess is None else self.guess[None, :]
        p = c + (1 - c) * s
        dp = (1 - c) * s * (1 - s)
        return (A[None, :] ** 2) * dp * dp / (p * (1 - p))


# ---------------------------------------------------------------- baselines

def item_mean(Y, W, prior=1.0):
    """Item correct rate among fitting entries, shrunk towards the global mean."""
    g = (W * Y).sum() / W.sum()
    return ((W * Y).sum(0) + prior * g) / (W.sum(0) + prior)


def model_mean(Y, W, prior=1.0):
    g = (W * Y).sum() / W.sum()
    return ((W * Y).sum(1) + prior * g) / (W.sum(1) + prior)


@dataclass
class LowRankLS:
    """Squared-loss low-rank completion with item and model intercepts:
    p_mq = mu_q + c_m + u_m^T v_q (iterative hard-impute with singular-value shrinkage)."""
    rank: int
    lam: float = 1.0
    ridge: float = 1.0
    mu: np.ndarray = None
    c: np.ndarray = None
    U: np.ndarray = None
    V: np.ndarray = None

    def fit(self, Y, W, n_iter=60, seed=0):
        Y = Y.astype(np.float64)
        W = W.astype(bool)
        self.mu = item_mean(Y, W)
        R = np.where(W, Y - self.mu[None, :], 0.0)
        self.c = R.sum(1) / (W.sum(1) + self.ridge)
        R = np.where(W, R - self.c[:, None], 0.0)
        Z = R.copy()
        for _ in range(n_iter):
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
            s = np.maximum(s[:self.rank] - self.lam, 0.0)
            low = (U[:, :self.rank] * s) @ Vt[:self.rank]
            Z_new = np.where(W, R, low)
            if np.abs(Z_new - Z).max() < 1e-5:
                Z = Z_new
                break
            Z = Z_new
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        s = np.maximum(s[:self.rank] - self.lam, 0.0)
        self.U = U[:, :self.rank] * np.sqrt(s)
        self.V = Vt[:self.rank].T * np.sqrt(s)
        return self

    def predict(self, U=None, c=None):
        U = self.U if U is None else U
        c = self.c if c is None else c
        return np.clip(self.mu[None, :] + c[:, None] + U @ self.V.T, EPS, 1 - EPS)

    def fold_in(self, Y, W):
        """Ridge estimate of (u_m, c_m) for new models from their observed entries."""
        Y = Y.astype(np.float64)
        W = W.astype(np.float64)
        R = W * (Y - self.mu[None, :])
        F = np.column_stack([self.V, np.ones(len(self.mu))])
        k = F.shape[1]
        out = np.zeros((Y.shape[0], k))
        for i in range(Y.shape[0]):
            H = F.T @ (F * W[i][:, None]) + self.ridge * np.eye(k)
            out[i] = np.linalg.solve(H, F.T @ R[i])
        return out[:, :-1], out[:, -1]


@dataclass
class KNN:
    """k-nearest-model predictor: average responses of the k reference models that agree
    most with the target model on its observed items, shrunk towards the item mean."""
    k: int
    Yref: np.ndarray = None
    mu: np.ndarray = None

    def fit(self, Yref, Wref):
        self.Yref = np.where(Wref, Yref, np.nan)
        self.mu = item_mean(Yref, Wref)
        return self

    def predict_for(self, Y, W, exclude_self=None):
        Yr = self.Yref
        Wr = ~np.isnan(Yr)
        Yr0 = np.nan_to_num(Yr)
        Yf = Y.astype(np.float64)
        Wf = W.astype(np.float64)
        # agreement rate on co-observed items
        agree = Wf * Yf @ (Wr * Yr0).T + Wf * (1 - Yf) @ (Wr * (1 - Yr0)).T
        co = Wf @ Wr.T.astype(np.float64)
        sim = agree / np.maximum(co, 1)
        if exclude_self is not None:
            sim[np.arange(len(exclude_self)), exclude_self] = -np.inf
        idx = np.argsort(-sim, axis=1)[:, :self.k]
        out = np.empty(Y.shape)
        for i in range(Y.shape[0]):
            nb = Yr[idx[i]]
            cnt = (~np.isnan(nb)).sum(0)
            out[i] = (np.nansum(nb, 0) + self.mu) / (cnt + 1)
        return np.clip(out, EPS, 1 - EPS)


# ---------------------------------------------------------------- metrics

def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    return float(((p - y) ** 2).mean())


def auc(y, p):
    from scipy.stats import rankdata
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
