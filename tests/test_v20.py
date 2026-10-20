"""v2.0: driver skill from history priors, comparable across teams."""
import json

import numpy as np

import purple_sectors.export as X
from helpers import laps_frame, posterior, write_laps
from purple_sectors.fit import prior_conflicts
from purple_sectors.prepare import build_design, driver_prior_arrays, load_push_laps, needs_driver_priors
from purple_sectors.simulate import draw_params

PRIORS = {"sigma_driver": 0.25, "sigma_drift": 0.05,
          "drivers": {"AAA": {"mean": -0.3, "sd": 0.05}, "BBB": {"mean": 0.1, "sd": 0.05}, "CCC": {"mean": -0.1, "sd": 0.06},
                      "DDD": {"mean": 0.2, "sd": 0.06}, "EEE": {"mean": 0.0, "sd": 0.08}}}       # FFF is a rookie


def test_prior_arrays_and_rookies():
    mean, sd, kind = driver_prior_arrays(["AAA", "FFF"], PRIORS)
    assert mean.tolist() == [-0.3, 0.0] and kind == ["history", "rookie"]
    assert np.isclose(sd[1], np.hypot(0.25, 0.05))                                # population prior, widened by drift


def test_v20_model_needs_priors_and_data_carries_them(tmp_path):
    assert needs_driver_priors("stan/quali_v20.stan") and not needs_driver_priors("stan/quali_v13.stan")
    des = build_design(load_push_laps(str(write_laps(tmp_path, laps_frame(3)))))
    data = des.stan_data(des.laps["y"].to_numpy(), priors=PRIORS)
    assert len(data["delta_prior_mean"]) == data["D"] == len(data["delta_prior_sd"])


def test_simulated_skill_follows_priors_without_centring(tmp_path):
    des = build_design(load_push_laps(str(write_laps(tmp_path, laps_frame(3)))))
    draws = np.array([draw_params(des, np.random.default_rng(k), v13=True, priors=PRIORS)["delta"] for k in range(400)])
    assert abs(draws[:, des.drivers.index("AAA")].mean() + 0.3) < 0.02            # follows its history prior
    assert abs(draws.mean(axis=1).mean()) > 0.01                                    # not forced to sum to zero


def test_conflicts_flag_only_big_moves():
    delta = np.random.default_rng(0).normal([-0.3, 0.1 + 0.4], 0.01, (500, 2))      # BBB moved 8 prior sds
    out = prior_conflicts(delta, ["AAA", "BBB"], np.array([-0.3, 0.1]), np.array([0.05, 0.05]))
    assert [c["driver"] for c in out] == ["BBB"] and out[0]["z"] > 2


def test_export_ranks_drivers_across_teams(tmp_path):
    D = 6
    path, df = posterior(tmp_path, extra=dict(delta_prior_mean=np.zeros(D), delta_prior_sd=np.full(D, 0.2),
                                              delta_prior_kind=np.array(["history"] * 5 + ["rookie"])))
    X.main(path, write_laps(tmp_path, df), tmp_path / "site", snapshot_dir=tmp_path / "snaps")
    soft = json.loads((tmp_path / "site" / "soft.json").read_text())
    assert soft["model"] == "v2.0" and soft["cross_team"]
    ranks = soft["drivers"]
    assert len(ranks) == D and ranks[0]["skill"][1] <= ranks[-1]["skill"][1]       # sorted fastest first
    assert abs(sum(r["p_fastest"] for r in ranks) - 1) < 0.01                         # each rounded to 3 decimals
    assert abs(sum(r["skill"][1] for r in ranks)) < 0.05                             # relative to the average driver
    draws = json.loads((tmp_path / "snaps" / "r04_draws.json").read_text())
    assert draws["cross_team"] and abs(np.mean([np.mean(v) for v in draws["drivers"].values()])) < 1e-3
