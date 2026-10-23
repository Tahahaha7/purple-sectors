"""Pace model tests: fake laps in, correct design and JSON out. No CmdStan, no network."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from purple_sectors.pace import data as D
from purple_sectors.pace import export as X
from purple_sectors.pace import simulate as S
from purple_sectors.pace.fit import gate

TEAMS = ["Alpha", "Beta", "Gamma"]
# season -> team -> drivers. MOV moves Alpha -> Gamma in 2025; NEW is a 2025 rookie.
GRID = {2024: {"Alpha": ["AAA", "MOV"], "Beta": ["BBB", "CCC"], "Gamma": ["DDD", "EEE"]},
        2025: {"Alpha": ["AAA", "NEW"], "Beta": ["BBB", "CCC"], "Gamma": ["DDD", "MOV"]}}


def fake_laps(tmp_path, rounds=3, laps_per_seg=3, seed=0):
    rng = np.random.default_rng(seed)
    files = []
    for season, grid in GRID.items():
        rows = []
        t0 = pd.Timestamp(f"{season}-03-01T14:00:00Z")
        for rnd in range(1, rounds + 1):
            for seg_i, seg in enumerate(["Q1", "Q2", "Q3"]):
                start = t0 + pd.Timedelta(days=14 * (rnd - 1), minutes=20 * seg_i)
                for team, drivers in grid.items():
                    for drv in drivers:
                        for k in range(laps_per_seg):
                            base = 90 + TEAMS.index(team) * 0.5 - seg_i * 0.3
                            rows.append(dict(season=season, round=rnd, event=f"GP{rnd}", location=f"Town{rnd}",
                                             segment=seg, driver=drv, name=f"Name {drv}", number=10 + k,
                                             team=team, lap_time=base + rng.normal(0, 0.2) + (3 if k == 2 and drv == "AAA" else 0),
                                             deleted=(k == 1 and drv == "BBB"), is_pit_out_lap=False, rainfall=0,
                                             compound="SOFT" if seg_i else "MEDIUM", tyre_life=k + 1,
                                             date_start=(start + pd.Timedelta(minutes=2 * k)).isoformat()))
        # one wet lap and one out-lap that must be dropped
        rows.append(dict(rows[0], rainfall=1.0))
        rows.append(dict(rows[0], is_pit_out_lap=True))
        p = tmp_path / f"quali_laps_{season}.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        files.append(p)
    return files


def test_push_filter_and_covariates(tmp_path):
    laps = D.load_laps(fake_laps(tmp_path))
    assert laps["rainfall"].eq(0).all() and not laps["is_pit_out_lap"].any()
    assert laps["deleted"].any(), "deleted laps are kept by default"
    assert not D.load_laps(fake_laps(tmp_path), keep_deleted=False)["deleted"].any()
    assert laps["gap_pct"].max() <= 7.0 and (laps["gap_pct"] > 3).any(), "the 3 s AAA laps stay under the 7% rule"
    assert set(laps["medium"].unique()) == {0.0, 1.0}
    assert np.allclose(laps.groupby(["season", "round", "segment"])["t"].mean(), 0)
    assert laps["tyre_age"].min() == 0 and laps["tyre_age"].max() == 2


def test_design_indices(tmp_path):
    des = D.build_design(D.load_laps(fake_laps(tmp_path)))
    assert des.seasons == [2024, 2025] and des.T == [3, 3] and des.R == [3, 3]
    assert des.D == 7 and des.E == 6 and des.A == 18
    assert des.P == 12, "a car is a team in one season, so 6 seats x 2 seasons = 12 pairs"
    assert des.season.min() == 1 and des.season.max() == 2 and des.rnd.max() == 3
    assert des.base.min() == 1 and des.base.max() == des.A and des.event.max() == des.E
    assert len(des.y_ref) == des.A and np.isfinite(des.y_ref).all()
    assert des.dt.shape == (2, 3) and des.dt[0, 0] == 0 and abs(des.dt[0, 1] - 2.0) < 1e-6
    assert abs(des.dt_bar - 2.0) < 1e-6
    sd = des.stan_data()
    assert sd["N"] == des.N and sd["TMAX"] == 3 and sd["RMAX"] == 3 and sd["prior_only"] == 0
    assert sd["present"].shape == (des.D, 2) and sd["y"].shape == (des.N,)
    mov = des.drivers.index("MOV")
    assert {(s, t) for d, s, t in des.pairs if d == mov} == {(0, 0), (1, 2)}


def test_rookie_and_young_flags(tmp_path):
    laps = D.load_laps(fake_laps(tmp_path))
    des = D.build_design(laps)                       # no history file: first seen after 2024 = rookie
    new, aaa = des.drivers.index("NEW"), des.drivers.index("AAA")
    assert des.rookie[new].tolist() == [0, 1] and des.young[new].tolist() == [0, 0]
    assert des.rookie[aaa].tolist() == [0, 0] and des.present[new].tolist() == [0, 1]
    des = D.build_design(laps, debuts={"AAA": 2023, "NEW": 2025})   # with history: AAA is in season two
    assert des.young[des.drivers.index("AAA")].tolist() == [1, 1]
    assert des.rookie[des.drivers.index("AAA")].tolist() == [0, 0]


def test_connectivity(tmp_path):
    des = D.build_design(D.load_laps(fake_laps(tmp_path)))
    net = D.connectivity(des)
    assert not net["connected"], "Beta never swaps a driver with anyone"
    assert sorted(map(len, net["components"])) == [2, 5]
    assert net["cars_per_driver"]["MOV"] == 2 and net["cars_per_driver"]["NEW"] == 1


def test_simulator_matches_model_structure(tmp_path):
    des = D.build_design(D.load_laps(fake_laps(tmp_path)))
    rng = np.random.default_rng(1)
    p = S.draw_params(des, rng, **S.TRUTH)
    assert p["car"].shape == (2, 3, 3) and p["skill"].shape == (des.D, 2) and p["fit"].shape == (des.P,)
    for s_ in range(2):                              # cars are centred on the field at every round
        assert np.allclose(p["car"][s_].sum(axis=0), 0) and np.allclose(p["trend"][s_].sum(axis=0), 0)
        on = des.present[:, s_] == 1               # skill is centred on that season's grid
        assert abs(p["skill"][on, s_].mean()) < 1e-9
    assert not np.allclose(p["car"], p["trend"]), "the per-round deviation sits on top of the trend"
    y = S.simulate_laps(des, p, rng)
    assert y.shape == (des.N,) and np.isfinite(y).all()
    mu = S.mu_of(des, p)
    assert (y - mu).mean() > 0, "laps only lose time to the ideal"


def test_exgauss_density_integrates_to_one():
    y = np.linspace(-5, 15, 200001)
    pdf = np.exp(S.exgauss_logpdf(y, 1.0, 0.4, 0.8))
    assert abs(np.trapezoid(pdf, y) - 1) < 1e-4
    assert abs((y * pdf * (y[1] - y[0])).sum() - 1.8) < 1e-3, "mean = mu + loss"
    pb = S.p_bad(np.array([0.9, 1.0, 1.2, 3.0, 6.0]), 1.0, 0.25, 0.3, 1.2, 0.1)
    assert (np.diff(pb) > 0).all() and pb[0] < 0.1 and pb[-1] > 0.9


def test_gate():
    good = dict(divergences=0, draws=4000, div_frac=0, max_rhat=1.005, min_ess_bulk=900, min_ess_tail=700,
                min_bfmi=0.8, treedepth_frac=0)
    assert gate(good) == "pass"
    assert gate(dict(good, divergences=2, div_frac=0.0005)) == "warn"
    assert gate(dict(good, max_rhat=1.03)) == "fail"
    assert gate(dict(good, min_ess_tail=150)) == "fail"
    assert gate(dict(good, min_bfmi=0.25)) == "warn"


def fake_posterior(des, ndraw=120, seed=2):
    rng = np.random.default_rng(seed)
    draws = [S.draw_params(des, rng, **S.TRUTH) for _ in range(ndraw)]
    post = {k: np.array([p[k] for p in draws]) for k in S.HYPERS + ["alpha", "gamma", "car", "trend", "skill", "fit"]}
    post["sigma_dev"] = np.sqrt(post["rho"]) * post["sigma_car_round"] / np.sqrt(des.dt_bar)
    post["sigma_nu"] = np.sqrt(1 - post["rho"]) * post["sigma_car_round"]
    post["loss_bad"] = post["loss_clean"] + post["loss_gap"]
    for k in ("season", "rnd", "team", "driver", "pair", "base", "event", "y"):
        post[k] = getattr(des, k)
    meta = dict(seasons=des.seasons, teams={str(s): des.teams[s] for s in des.seasons}, drivers=des.drivers,
                events=[dict(season=int(s), round=int(r), name=f"GP{r}", location="", date="2025-01-01")
                        for s, r in zip(des.events["season"], des.events["round"])],
                pairs=[list(p) for p in des.pairs], rookie=des.rookie.tolist(), young=des.young.tolist(),
                names={d: dict(name=f"Name {d}", number=1) for d in des.drivers}, connectivity=D.connectivity(des))
    return post, meta, dict(status="pass")


def test_export_grid_and_rankings(tmp_path):
    des = D.build_design(D.load_laps(fake_laps(tmp_path)))
    out = X.build(*fake_posterior(des))
    assert out["season"] == 2025 and out["round"] == 3 and out["status"] == "pass"
    assert len(out["cars"]) == 3 and len(out["drivers"]) == 6 and len(out["teammates"]) == 3
    assert abs(sum(c["p_fastest"] for c in out["cars"]) - 1) < 2e-3   # rounded to 3 decimals
    assert abs(sum(d["p_fastest"] for d in out["drivers"]) - 1) < 4e-3
    assert [c["pace"][1] for c in out["cars"]] == sorted(c["pace"][1] for c in out["cars"])
    assert all(c["last"][0] >= 0 and c["gap"][0] >= 0 for c in out["cars"])
    skills = [d["skill"][1] for d in out["drivers"]]
    assert skills == sorted(skills)
    mov = next(d for d in out["drivers"] if d["code"] == "MOV")
    assert mov["moved"] and mov["teams_driven"] == 2 and mov["team"] == "Gamma"
    assert not next(d for d in out["drivers"] if d["code"] == "BBB")["moved"]
    assert next(d for d in out["drivers"] if d["code"] == "NEW")["rookie"]
    g = out["grid"]
    assert len(g["cells"]) == 6 and all(len(row) == 3 for row in g["cells"])
    for i, row in enumerate(g["cells"]):
        seats = [c for c in row if c["seat"]]
        assert len(seats) == 1 and seats[0]["seen"] and seats[0]["laps"] > 0
        for c in row:
            assert 1 <= c["pos"][0] <= c["pos"][1] <= c["pos"][2] <= 6
            assert c["pace"][0] <= c["pace"][1] <= c["pace"][2]
    mates = {(m["a"], m["b"]) for m in out["teammates"]}
    assert all(m["p_a_ahead"] >= 0.5 for m in out["teammates"]) and ("AAA", "NEW") in mates or ("NEW", "AAA") in mates
    json.dumps(out)  # must be serialisable


def test_short_names():
    assert X.short("Red Bull Racing") == "RBR" and X.short("Kick Sauber") == "SAU" and X.short("Zed Motors") == "ZED"
