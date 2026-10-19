"""Generative model in NumPy, mirroring stan/quali.stan line by line.

Used for (1) prior predictive checks and (2) parameter recovery:
simulate a season from known parameters, fit Stan, check we get them back.
"""
import numpy as np
import pandas as pd

from .prepare import Design

PRIORS = dict(  # (mean, sd) for normals; sd only for half-normals
    m=(-0.5, 0.3), sigma_kappa=0.3, sigma_car=1.2, sigma_dev=0.25, sigma_nu=0.5,
    sigma_driver=0.3, beta_m=(0.7, 0.5), beta_a=(0.05, 0.05), sigma=0.5, loss=0.5,
    sigma_total=1.2,  # Model B only
    mu_gamma=(-0.3, 0.3), sigma_gamma=0.2,  # v1.1+
    pi_bad=(2, 6), loss_clean=0.3, loss_gap=1.5,  # v1.2
    sigma_car_event=0.6, rho_dev=(2, 4))  # v1.3


def _sum_to_zero(rng, scale, size):
    """Draws matching Stan's sum_to_zero_vector ~ normal(0, scale * sqrt(K/(K-1)))."""
    k = size[-1]
    z = rng.normal(0, scale * np.sqrt(k / (k - 1)), size)
    return z - z.mean(axis=-1, keepdims=True)


def draw_params(des: Design, rng, share_prior: bool = False, v11: bool = False,
                v12: bool = False, v13: bool = False, **fixed) -> dict:
    """Draw hyperparameters from the priors (unless fixed), then all latent effects.
    share_prior=True uses Model B: rho ~ Beta(1,1) splits sigma_total into car and driver."""
    hn = lambda s: abs(rng.normal(0, s))
    p = dict(
        m=rng.normal(*PRIORS["m"], size=2), sigma_kappa=hn(PRIORS["sigma_kappa"]),
        sigma_car=hn(PRIORS["sigma_car"]), sigma_dev=hn(PRIORS["sigma_dev"]),
        sigma_nu=hn(PRIORS["sigma_nu"]), sigma_driver=hn(PRIORS["sigma_driver"]),
        beta_m=rng.normal(*PRIORS["beta_m"]), beta_a=rng.normal(*PRIORS["beta_a"]),
        sigma=hn(PRIORS["sigma"]), loss=max(hn(PRIORS["loss"]), 0.01))
    if v13:          # development vs track fit as (total, share), like Model B does for car vs driver
        dt_bar = float(np.mean(des.dt[1:])) if des.E > 1 else 1.0
        tot = fixed.get("sigma_car_event", hn(PRIORS["sigma_car_event"]))
        rho = fixed.get("rho_dev", rng.beta(*PRIORS["rho_dev"]))
        p.update(sigma_dev=np.sqrt(rho / dt_bar) * tot, sigma_nu=np.sqrt(1 - rho) * tot, sigma_car_event=tot, rho_dev=rho)
        v12 = True
    if share_prior:
        rho, tot = rng.uniform(), hn(PRIORS["sigma_total"])
        p.update(sigma_car=np.sqrt(rho) * tot, sigma_driver=np.sqrt(1 - rho) * tot)
    p.update(fixed)

    E, C, D = des.E, des.C, des.D
    k = p["m"] + p["sigma_kappa"] * rng.normal(size=(E, 2))
    p["tau"] = np.c_[np.zeros(E), k[:, 0], k[:, 0] + k[:, 1]]

    z = _sum_to_zero(rng, 1.0, (E, C))
    theta = np.empty((E, C))
    theta[0] = p["sigma_car"] * z[0]
    for e in range(1, E):
        theta[e] = theta[e - 1] + p["sigma_dev"] * np.sqrt(des.dt[e]) * z[e]
    p["theta"] = theta
    p["nu"] = p["sigma_nu"] * _sum_to_zero(rng, 1.0, (E, C))
    p["delta"] = p["sigma_driver"] * _sum_to_zero(rng, 1.0, (D,))
    if v12:
        v11 = True
        p.setdefault("pi_bad", rng.beta(*PRIORS["pi_bad"]))
        p.setdefault("loss_clean", max(hn(PRIORS["loss_clean"]), 0.01))
        p.setdefault("loss_gap", hn(PRIORS["loss_gap"]))
    if v11:
        p.setdefault("mu_gamma", rng.normal(*PRIORS["mu_gamma"]))
        p.setdefault("sigma_gamma", hn(PRIORS["sigma_gamma"]))
        p["gamma"] = p["mu_gamma"] + p["sigma_gamma"] * rng.normal(size=E)
    p["alpha"] = 100 * np.log(des.laps.groupby("round")["lap_time"].median()
                              .reindex(des.events).to_numpy())
    return p


def linear_predictor(des: Design, p: dict) -> np.ndarray:
    e, c = des.event, des.car
    return (p["alpha"][e] + p["tau"][e, des.seg] + p["theta"][e, c] + p["nu"][e, c]
            + p["delta"][des.driver] + p["beta_m"] * des.medium + p["beta_a"] * des.tyre_age
            + (p["gamma"][e] * des.t_seg if "gamma" in p else 0))


def simulate_laps(des: Design, p: dict, rng) -> np.ndarray:
    """y = mu + Normal(0, sigma) + Exponential(mean = loss)."""
    mu = linear_predictor(des, p)
    if "pi_bad" in p:                                   # v1.2 two-component execution
        bad = rng.uniform(size=des.N) < p["pi_bad"]
        loss = np.where(bad, p["loss_clean"] + p["loss_gap"], p["loss_clean"])
    else:
        loss = p["loss"]
    return mu + rng.normal(0, p["sigma"], des.N) + rng.exponential(loss, des.N)


def summaries(des: Design, y: np.ndarray) -> dict:
    """Statistics we can compare between simulated and real seasons (all in %)."""
    df = pd.DataFrame(dict(y=y, e=des.event, c=des.car, d=des.driver, s=des.seg))
    best_car = df.groupby(["e", "c"]).y.min()
    spread = (best_car.groupby("e").max() - best_car.groupby("e").min()).median()

    best_drv = df.groupby(["e", "c", "d"]).y.min().reset_index()
    pairs = best_drv.groupby(["e", "c"]).y.agg(["min", "max", "size"])
    mate = (pairs.loc[pairs["size"] == 2, "max"] - pairs.loc[pairs["size"] == 2, "min"]).median()

    seg_best = df.groupby(["e", "d", "s"]).y.min().unstack()
    evo = (seg_best[1] - seg_best[0]).median()

    g = df.groupby(["e", "s", "d"]).y
    dev = (df.y - g.transform("mean"))[g.transform("size") >= 3]
    skew = ((dev - dev.mean()) ** 3).mean() / dev.std() ** 3
    kurt = ((dev - dev.mean()) ** 4).mean() / dev.var() ** 2 - 3

    return dict(field_spread=spread, teammate_gap=mate, q1_q2_step=evo, lap_skew=skew,
                lap_kurt=kurt)
