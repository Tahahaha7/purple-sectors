"""PSIS-LOO (Vehtari, Gelman & Gabry 2017) in plain NumPy, for comparing model versions."""
import numpy as np


def _gpdfit(x):
    n = len(x); x = np.sort(x); prior_bs, prior_k = 3, 10
    m = 30 + int(n ** 0.5)
    b = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b /= prior_bs * x[int(n / 4 + 0.5) - 1]; b += 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    ls = n * (np.log(-(b / k)) - k - 1)
    w = 1 / np.exp(ls - ls[:, None]).sum(axis=1)
    ok = w >= 10 * np.finfo(float).eps; w = w[ok] / w[ok].sum()
    bp = (b[ok] * w).sum(); kp = np.log1p(-bp * x).mean()
    sigma = -kp / bp
    return (n * kp + prior_k * 0.5) / (n + prior_k), sigma


def psis_loo(log_lik: np.ndarray):
    """log_lik: draws x observations. Returns pointwise elpd and Pareto k."""
    log_lik = np.asarray(log_lik, dtype=float)
    S, N = log_lik.shape; M = int(min(0.2 * S, 3 * np.sqrt(S)))
    elpd, khat = np.empty(N), np.empty(N)
    for i in range(N):
        lw = -log_lik[:, i]; lw = lw - lw.max(); order = np.argsort(lw); tail = order[-M:]
        cut = lw[order[-M - 1]]; x = np.exp(lw[tail]) - np.exp(cut)
        k, sigma = _gpdfit(x)
        if np.isfinite(k):
            p = (np.arange(1, M + 1) - 0.5) / M
            q = sigma / k * ((1 - p) ** (-k) - 1) if abs(k) > 1e-12 else -sigma * np.log1p(-p)
            lw[tail] = np.minimum(np.log(q + np.exp(cut)), 0)
        khat[i] = k
        lw -= np.logaddexp.reduce(lw)
        elpd[i] = np.logaddexp.reduce(lw + log_lik[:, i])
    return elpd, khat


def compare(ll_a: np.ndarray, ll_b: np.ndarray) -> dict:
    """elpd of each model and the difference b - a with its standard error."""
    ea, ka = psis_loo(ll_a); eb, kb = psis_loo(ll_b); d = eb - ea
    return {"elpd_a": float(ea.sum()), "elpd_b": float(eb.sum()), "diff": float(d.sum()),
            "se": float(np.sqrt(len(d) * d.var())), "k_bad_a": int((ka > 0.7).sum()), "k_bad_b": int((kb > 0.7).sum())}
