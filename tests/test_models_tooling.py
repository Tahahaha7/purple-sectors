import json
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import laps_frame, write_laps
from purple_sectors.loo import compare, psis_loo
from purple_sectors.prepare import build_design, load_push_laps
from purple_sectors.recovery import flags_for
from purple_sectors.simulate import draw_params


def test_psis_loo_matches_exact_loo_for_a_normal_model():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 40); mu = rng.normal(y.mean(), 1 / np.sqrt(40), 4000)       # posterior of the mean
    ll = -0.5 * np.log(2 * np.pi) - 0.5 * (y[None, :] - mu[:, None]) ** 2
    elpd, k = psis_loo(ll)
    exact = [(-0.5 * np.log(2 * np.pi * (1 + 1 / 39)) - 0.5 * (y[i] - np.delete(y, i).mean()) ** 2 / (1 + 1 / 39)) for i in range(40)]
    assert abs(elpd.sum() - sum(exact)) < 0.5 and (k < 0.7).all()


def test_compare_prefers_the_better_model():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 60); good = rng.normal(0, 0.1, 2000); bad = rng.normal(2, 0.1, 2000)
    ll = lambda mu: -0.5 * (y[None, :] - mu[:, None]) ** 2
    out = compare(ll(bad), ll(good))
    assert out["diff"] > 2 * out["se"]


def test_v13_draws_keep_names_and_fixed_values(tmp_path):
    des = build_design(load_push_laps(str(write_laps(tmp_path, laps_frame(4)))))
    p = draw_params(des, np.random.default_rng(2), v13=True, sigma_car_event=0.3, rho_dev=0.1)
    dt_bar = float(np.mean(des.dt[1:]))
    assert np.isclose(p["sigma_dev"], np.sqrt(0.1 / dt_bar) * 0.3) and np.isclose(p["sigma_nu"], np.sqrt(0.9) * 0.3)
    assert "pi_bad" in p and "gamma" in p                                          # v1.3 includes the v1.2 mixture


def test_model_version_flags():
    assert flags_for(Path("stan/quali_v13.stan")) == {"v13": True}
    assert flags_for(Path("stan/quali_v12.stan")) == {"v12": True}
    assert flags_for(Path("stan/quali.stan")) == {}
