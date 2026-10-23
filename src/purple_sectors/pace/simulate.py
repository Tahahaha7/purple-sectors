"""NumPy mirror of stan/pace.stan.

Used three ways: prior predictive checks, parameter recovery (simulate a fake history on
the real design, fit, check the truth is recovered) and, after a fit, the probability that
each lap was compromised. Keep this file and the Stan file in step.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, logsumexp

from .data import Design

# Prior scales, exactly as in pace.stan. Pairs are (mean, sd); singles are half-normal sds.
PRIORS = dict(
    mu_gamma=(0.0, 0.5), sigma_gamma=0.3,
    sigma_car=1.5, sigma_car_round=0.6, rho=(2, 4),
    sigma_driver=0.3, sigma_drift=0.1, sigma_young=0.2, mu_rookie=(0.0, 0.3), mu_young=(0.0, 0.15),
    sigma_fit=0.2, beta_medium=(0.5, 0.5), beta_age=(0.0, 0.1),
    sigma=0.5, loss_clean=0.5, loss_gap=1.0, pi_bad=(2, 12),
    sigma_best=0.5, nu_best=(2, 0.1),        # best laps (earlier seasons): Student-t noise and tail weight
)

# A realistic "truth" for recovery checks (% of lap time).
TRUTH = dict(
    mu_gamma=-0.3, sigma_gamma=0.15, sigma_car=1.0, sigma_car_round=0.33, rho=0.08,
    sigma_driver=0.25, sigma_drift=0.06, sigma_young=0.12, mu_rookie=0.3, mu_young=-0.1,
    sigma_fit=0.1, beta_medium=0.6, beta_age=0.04, sigma=0.25, loss_clean=0.3, loss_gap=0.9, pi_bad=0.1,
    sigma_best=0.35, nu_best=8.0,
)

HYPERS = list(PRIORS)


def draw_hypers(rng, **fixed) -> dict:
    p = {}
    for k, spec in PRIORS.items():
        if k in fixed:
            p[k] = float(fixed[k])
        elif k in ("rho", "pi_bad"):
            p[k] = float(rng.beta(*spec))
        elif k == "nu_best":
            p[k] = float(np.clip(rng.gamma(spec[0], 1 / spec[1]), 1, 100))
        elif isinstance(spec, tuple):
            p[k] = float(rng.normal(*spec))
        else:
            p[k] = float(abs(rng.normal(0, spec)))
    return p


def draw_params(des: Design, rng, **fixed) -> dict:
    """Hyperparameters from the priors (unless fixed), then every latent effect."""
    p = draw_hypers(rng, **fixed)
    S, TMAX, RMAX, D = des.S, des.TMAX, des.RMAX, des.D
    p["alpha"] = des.y_ref + 2 * rng.normal(size=des.A)
    p["gamma"] = p["mu_gamma"] + p["sigma_gamma"] * rng.normal(size=des.E)

    sigma_dev = np.sqrt(p["rho"]) * p["sigma_car_round"] / np.sqrt(des.dt_bar)
    sigma_nu = np.sqrt(1 - p["rho"]) * p["sigma_car_round"]
    car = np.zeros((S, TMAX, RMAX))
    trend = np.zeros((S, TMAX, RMAX))
    for s, (T, R) in enumerate(zip(des.T, des.R)):
        walk = p["sigma_car"] * rng.normal(size=T)
        for r in range(R):
            if r:
                walk = walk + sigma_dev * np.sqrt(des.dt[s, r]) * rng.normal(size=T)
            c = walk + sigma_nu * rng.normal(size=T)
            trend[s, :T, r] = walk - walk.mean()
            car[s, :T, r] = c - c.mean()
    p["car"] = car
    p["trend"] = trend

    walk = np.zeros((D, S))
    for d in range(D):
        for s in range(S):
            if s == 0 or des.rookie[d, s]:
                walk[d, s] = des.rookie[d, s] * p["mu_rookie"] + p["sigma_driver"] * rng.normal()
            else:
                sd = np.sqrt(p["sigma_drift"] ** 2 + des.young[d, s] * p["sigma_young"] ** 2)
                walk[d, s] = walk[d, s - 1] + des.young[d, s] * p["mu_young"] + sd * rng.normal()
    skill = walk.copy()
    for s in range(S):
        on = des.present[:, s] == 1
        skill[:, s] -= walk[on, s].mean()
    p["skill"] = skill
    p["fit"] = p["sigma_fit"] * rng.normal(size=des.P)
    return p


def mu_of(des: Design, p: dict) -> np.ndarray:
    """The modelled lap for every lap in the design, given one set of parameters."""
    s, r, tm, d = des.season - 1, des.rnd - 1, des.team - 1, des.driver - 1
    return (p["alpha"][des.base - 1] + p["gamma"][des.event - 1] * des.t
            + p["car"][s, tm, r] + p["skill"][d, s] + p["fit"][des.pair - 1]
            + p["beta_medium"] * des.medium + p["beta_age"] * des.tyre_age)


def mu2_of(des: Design, p: dict) -> np.ndarray:
    """The modelled best lap for every earlier-season best lap in the design."""
    s, r, tm, d = des.season2 - 1, des.rnd2 - 1, des.team2 - 1, des.driver2 - 1
    return p["alpha"][des.base2 - 1] + p["car"][s, tm, r] + p["skill"][d, s] + p["fit"][des.pair2 - 1]


def simulate_best(des: Design, p: dict, rng) -> np.ndarray:
    if des.N2 == 0:
        return np.zeros(0)
    return mu2_of(des, p) + p["sigma_best"] * rng.standard_t(p["nu_best"], des.N2)


def simulate_laps(des: Design, p: dict, rng) -> np.ndarray:
    mu = mu_of(des, p)
    bad = rng.random(des.N) < p["pi_bad"]
    loss = np.where(bad, p["loss_clean"] + p["loss_gap"], p["loss_clean"])
    return mu + rng.normal(0, p["sigma"], des.N) + rng.exponential(loss)


def exgauss_logpdf(y, mu, sigma, loss):
    """log density of Normal(mu, sigma) + Exponential(mean = loss)."""
    lam = 1.0 / loss
    return np.log(lam) + lam * (mu - y) + 0.5 * lam ** 2 * sigma ** 2 + log_ndtr((y - mu) / sigma - lam * sigma)


def lap_logpdf(y, mu, sigma, loss_clean, loss_bad, pi_bad):
    good = np.log1p(-pi_bad) + exgauss_logpdf(y, mu, sigma, loss_clean)
    bad = np.log(pi_bad) + exgauss_logpdf(y, mu, sigma, loss_bad)
    return logsumexp(np.stack([good, bad]), axis=0)


def p_bad(y, mu, sigma, loss_clean, loss_bad, pi_bad):
    """Probability each lap came from the compromised component."""
    good = np.log1p(-pi_bad) + exgauss_logpdf(y, mu, sigma, loss_clean)
    bad = np.log(pi_bad) + exgauss_logpdf(y, mu, sigma, loss_bad)
    return np.exp(bad - np.logaddexp(good, bad))


def params_from_draw(post: dict, i: int) -> dict:
    """One posterior draw in the same shape draw_params returns."""
    p = {k: float(post[k][i]) for k in HYPERS if k in post}
    p["loss_bad"] = float(post["loss_bad"][i]) if "loss_bad" in post else p["loss_clean"] + p["loss_gap"]
    for k in ("alpha", "gamma", "car", "trend", "skill", "fit"):
        if k in post:
            p[k] = np.asarray(post[k][i], dtype=float)
    return p
