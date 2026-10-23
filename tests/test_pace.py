"""Pace model tests: fake laps in, correct design and JSON out. No CmdStan, no network."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from purple_sectors.pace import data as D
from purple_sectors.pace import export as X
from purple_sectors.pace import forecast as FC
from purple_sectors.pace import simulate as S
from purple_sectors.pace import fit as PF
from purple_sectors.pace.fit import gate

TEAMS = ["Alpha", "Beta", "Gamma"]
# season -> team -> drivers. MOV moves Alpha -> Gamma in 2025; NEW is a 2025 rookie.
GRID = {2024: {"Alpha": ["AAA", "MOV"], "Beta": ["BBB", "CCC"], "Gamma": ["DDD", "EEE"]},
        2025: {"Alpha": ["AAA", "NEW"], "Beta": ["BBB", "CCC"], "Gamma": ["DDD", "MOV"]}}


def fake_history(tmp_path, season=2022):
    """A Jolpica-style best-lap file: BBB drove for Gamma in 2022, which links Beta to everyone else."""
    rows = []
    grid = {"Alpha": ["AAA", "MOV"], "Beta": ["XXX", "CCC"], "Gamma": ["DDD", "BBB"]}
    for rnd in range(1, 4):
        for team, drivers in grid.items():
            for i, drv in enumerate(drivers):
                base = 90 + ["Alpha", "Beta", "Gamma"].index(team) * 0.5 + 0.1 * i
                rows.append(dict(season=season, round=rnd, race=f"Old GP{rnd}", circuit=f"c{rnd}", location=f"Oldtown{rnd}",
                                 date=f"{season}-0{3 + rnd}-01", driver=drv, driver_id=drv.lower(), name=f"Name {drv}",
                                 number=20 + i, team_id=team.lower(), team=team, position=i + 1,
                                 q1=base, q2=base - 0.3 if rnd < 3 else None, q3=base - 0.6 if i == 0 else None))
    rows.append(dict(rows[0], driver="SLO", driver_id="slo", name="Name SLO", q1=rows[0]["q1"] * 1.2, q2=None, q3=None))  # beyond 107%
    path = tmp_path / "history_quali_2022_2025.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


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


def test_best_laps_join_the_design(tmp_path):
    laps = D.load_laps(fake_laps(tmp_path))
    best = D.load_best_laps(fake_history(tmp_path), [2022])
    assert "SLO" not in best["driver"].values, "the 107% cut applies to best laps too"
    assert len(best) == 3 * 6 + 2 * 6 + 3 * 3, "q1 every round, q2 in two rounds, q3 for the first driver"
    des = D.build_design(laps, None, best)
    assert des.seasons == [2022, 2024, 2025] and des.T == [3, 3, 3] and des.R == [3, 3, 3]
    assert des.N2 == len(best) and des.N == len(laps)
    assert des.season2.min() == 1 and des.season2.max() == 1 and des.base2.min() >= 1 and des.base2.max() <= des.A
    assert des.pair2.max() <= des.P and des.driver2.max() <= des.D and des.team2.max() <= 3
    assert np.isfinite(des.y_ref).all(), "2022 baselines take their reference from the best laps"
    assert D.connectivity(des)["connected"], "BBB's 2022 season at Gamma links Beta to the rest"
    sd = des.stan_data()
    assert sd["N2"] == des.N2 and sd["y2"].shape == (des.N2,) and sd["present"].shape == (des.D, 3)
    xxx = des.drivers.index("XXX")
    assert des.present[xxx].tolist() == [1, 0, 0] and des.rookie[xxx].tolist() == [0, 0, 0]
    rng = np.random.default_rng(3)
    p = S.draw_params(des, rng, **S.TRUTH)
    y2 = S.simulate_best(des, p, rng)
    assert y2.shape == (des.N2,) and np.isfinite(y2).all() and abs((y2 - S.mu2_of(des, p)).mean()) < 0.2
    assert D.design_from_files(tmp_path, [2024, 2025], history_csv=tmp_path / "history_quali_2022_2025.csv").N2 == des.N2
    assert D.design_from_files(tmp_path, [2024, 2025], history_csv=tmp_path / "missing.csv").N2 == 0


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


class FakeFit:
    """Duck-typed CmdStanMCMC: just enough for diagnostics() and posterior_dict()."""

    def __init__(self, post, n=60, chains=2, seed=0):
        rng = np.random.default_rng(seed)
        self.post = {k: np.asarray(v) for k, v in post.items()}
        self.chains, self.num_draws_sampling = chains, n
        self.column_names = ("lp__", "accept_stat__", "divergent__", "treedepth__", "energy__", "sigma", "skill[1,1]")
        self._draws = rng.normal(size=(n, chains, len(self.column_names)))

    def stan_variable(self, name):
        return self.post[name]

    def draws(self, inc_warmup=False):
        return self._draws

    def method_variables(self):
        n, m = self.num_draws_sampling, self.chains
        return {"divergent__": np.zeros((n, m)), "treedepth__": np.full((n, m), 4), "energy__": np.random.default_rng(1).normal(size=(n, m))}


def test_posterior_dict_and_diagnostics_with_a_fake_fit(tmp_path):
    des = D.build_design(D.load_laps(fake_laps(tmp_path)), None, D.load_best_laps(fake_history(tmp_path), [2022]))
    post, meta, diag = fake_posterior(des)
    out = PF.posterior_dict(FakeFit(post, n=60), des, keep=40, seed=0)
    assert out["skill"].shape[0] == 40 and out["p_bad_mean"].shape == (des.N,) and np.isfinite(out["p_bad_mean"]).all()
    assert set(["car", "trend", "fit", "alpha", "gamma", "loss_bad", "sigma_best", "nu_best"]) <= set(out)
    meta2 = json.loads(str(out["meta_json"]))
    assert meta2["best_laps"] == des.N2 and "XXX" in meta2["names"] and meta2["names"]["XXX"]["name"] == "Name XXX"
    d = PF.diagnostics(FakeFit(post), 10)
    assert set(["divergences", "max_rhat", "min_ess_bulk", "min_ess_tail", "min_bfmi", "treedepth_frac", "worst"]) <= set(d)
    assert d["divergences"] == 0 and d["treedepth_frac"] == 0 and d["max_rhat"] < 1.2


def test_gate():
    good = dict(divergences=0, draws=4000, div_frac=0, max_rhat=1.005, min_ess_bulk=900, min_ess_tail=700,
                min_bfmi=0.8, treedepth_frac=0)
    assert gate(good) == "pass"
    assert gate(dict(good, divergences=2, div_frac=0.0005)) == "warn"
    assert gate(dict(good, max_rhat=1.03)) == "fail"
    assert gate(dict(good, min_ess_tail=150)) == "fail"
    assert gate(dict(good, min_bfmi=0.25)) == "warn"


def fake_posterior(des, ndraw=120, seed=2):
    """A posterior in the layout fit.posterior_dict writes, drawn from the prior truth."""
    rng = np.random.default_rng(seed)
    draws = [S.draw_params(des, rng, **S.TRUTH) for _ in range(ndraw)]
    post = {k: np.array([p[k] for p in draws]) for k in S.HYPERS + ["alpha", "gamma", "car", "trend", "skill", "fit"]}
    post["sigma_dev"] = np.sqrt(post["rho"]) * post["sigma_car_round"] / np.sqrt(des.dt_bar)
    post["sigma_nu"] = np.sqrt(1 - post["rho"]) * post["sigma_car_round"]
    post["loss_bad"] = post["loss_clean"] + post["loss_gap"]
    for k in ("season", "rnd", "team", "driver", "pair", "base", "event", "y", "t", "medium", "tyre_age"):
        post[k] = getattr(des, k)
    post["p_bad_mean"] = np.full(des.N, 0.1)
    ev = des.events
    meta = dict(seasons=des.seasons, teams={str(x): des.teams[x] for x in des.seasons}, drivers=des.drivers,
                events=[dict(season=int(s_), round=int(r), r=int(k), e=int(e), date=str(d)[:10], name=f"GP{r}", location=f"Town{r}")
                        for s_, r, k, e, d in zip(ev["season"], ev["round"], ev["r"], ev["e"], ev["date"])],
                pairs=[list(p) for p in des.pairs], bases=[list(b) for b in des.bases], present=des.present.tolist(),
                rookie=des.rookie.tolist(), young=des.young.tolist(),
                names={d: dict(name=f"Name {d}", number=i + 1) for i, d in enumerate(des.drivers)}, connectivity=D.connectivity(des))
    diag = dict(status="pass", divergences=0, draws=4000, max_rhat=1.003, worst="skill[1,1]", min_ess_bulk=900, min_ess_tail=800)
    return post, meta, diag


def view(tmp_path):
    des = D.build_design(D.load_laps(fake_laps(tmp_path)))
    post, meta, diag = fake_posterior(des)
    return X.season_view(post, meta, diag), post, meta, diag


def test_season_view(tmp_path):
    V, post, meta, diag = view(tmp_path)
    assert V["season"] == 2025 and V["R"] == 3 and V["T"] == 3 and V["rounds"] == [1, 2, 3]
    assert len(V["drivers"]) == 6 and "EEE" not in V["drivers"], "EEE left after 2024"
    assert V["theta"].shape == (120, 3, 3) and V["delta"].shape == (120, 6) and V["fit_seat"].shape == (120, 6)
    assert np.allclose(V["theta"].sum(axis=2), 0) and np.allclose(V["pace"].sum(axis=2), 0)
    new = V["drivers"].index("NEW")
    assert not V["has_prev"][new] and V["has_prev"][V["drivers"].index("MOV")]
    assert V["tau"].shape == (120, 3, 3) and (V["tau"][:, :, 0] == 0).all() and (V["tau"][:, :, 1] < 0).mean() > 0.5
    assert V["hypers"]["m"].shape == (120, 2) and set(V["fits"]["MOV"]) == {"Gamma"}
    assert len(V["lap_y"]) == V["laps_season"] == (post["season"] == 2).sum()


def test_soft_snapshot_and_draws(tmp_path):
    V = view(tmp_path)[0]
    soft = X.soft(V)
    assert soft["model"] == "pace" and soft["round"] == 3 and soft["cross_team"] and soft["status"] == "pass"
    assert [c["med"] for c in soft["cars"]] == sorted(c["med"] for c in soft["cars"])
    skills = [d["skill"][1] for d in soft["drivers"]]
    assert skills == sorted(skills) and len(soft["drivers"]) == 6
    mov = next(d for d in soft["drivers"] if d["code"] == "MOV")
    assert mov["history"] and mov["moved_teams"] and mov["team"] == "Gamma"
    assert not next(d for d in soft["drivers"] if d["code"] == "NEW")["history"]
    assert len(soft["pairs"]) == 3 and all(p["p"] >= 0.5 and len(p["dots"]) == 20 for p in soft["pairs"])
    assert all(len(v) == 96 for v in soft["draws"].values())
    snap = X.snapshot(V)
    assert len(snap["cars"]) == 3 and len(snap["cars"][0]["pace"]) == 21 and len(snap["drivers"]) == 6
    assert len(snap["events"]) == 3 and snap["events"][0]["gamma"] and "density" in snap["teammates"][0]
    assert snap["diagnostics"]["passed"] and snap["diagnostics"]["status"] == "pass" and "sigma_fit" in snap["params"]
    dr = X.draws_snapshot(V, 50)
    assert dr["n"] == 50 and len(dr["cars"]["Alpha"]) == 50 and len(dr["drivers"]["MOV"]) == 50
    assert set(dr["fit"]["MOV"]) == {"Gamma"} and len(dr["sigma_fit"]) == 50
    json.dumps([soft, snap, dr])


def test_grid(tmp_path):
    V = view(tmp_path)[0]
    g = X.grid(V)["grid"]
    assert len(g["cells"]) == 6 and all(len(row) == 3 for row in g["cells"])
    for row in g["cells"]:
        seats = [c for c in row if c["seat"]]
        assert len(seats) == 1 and seats[0]["seen"]
        for c in row:
            assert 1 <= c["pos"][0] <= c["pos"][1] <= c["pos"][2] <= 6 and c["pace"][0] <= c["pace"][1] <= c["pace"][2]
    assert abs(sum(row[j]["p_first"] for row in g["cells"] for j in range(3) if row[j]["seat"]) - 1) < 1e-2


def test_forecast(tmp_path):
    V = view(tmp_path)[0]
    F, pos = FC.simulate(V, weeks=2.0, n=80)
    assert pos.shape == (80, 6) and all(sorted(row) == list(range(1, 7)) for row in pos)
    fc = FC.forecast(V, {"round": 4, "name": "Next GP", "date": "2025-05-01"}, weeks=2.0, n=80)
    assert abs(sum(r["p_pole"] for r in fc["drivers"]) - 1) < 1e-6 and fc["made_after_round"] == 3
    assert all(r["pos_lo"] <= r["pos_median"] <= r["pos_hi"] for r in fc["drivers"])
    laps = pd.read_csv(fake_laps(tmp_path)[-1])
    sched = tmp_path / "schedule_2025.json"
    sched.write_text(json.dumps([{"round": 4, "name": "Next GP", "location": "X", "date": "2025-04-12T12:00:00+00:00"}]))
    tgt, weeks = FC.next_target(V, laps, sched)
    assert tgt["round"] == 4 and 1.5 < weeks < 2.5
    order = FC.classification(laps, 3)
    assert len(order) == 6 and order[0] in laps["driver"].unique()
    fcd = tmp_path / "forecasts"; fcd.mkdir(); (fcd / "r04.json").write_text(json.dumps(fc))
    card = FC.scorecard(fcd, laps)
    assert card["latest"] == "r04.json" and (fcd / "scorecard.json").exists()


def test_export_main_writes_everything(tmp_path):
    des = D.build_design(D.load_laps(fake_laps(tmp_path)))
    post, meta, diag = fake_posterior(des)
    post["meta_json"] = np.array(json.dumps(meta)); post["diag_json"] = np.array(json.dumps(diag))
    npz = tmp_path / "post.npz"; np.savez(npz, **post)
    laps = fake_laps(tmp_path)[-1]
    out = tmp_path / "site"
    X.main(npz, laps, out, snapshot_dir=out / "snapshots")
    for f in ["soft.json", "pace.json", "snapshots/r03.json", "snapshots/r03_draws.json", "snapshots/index.json",
              "forecasts/scorecard.json"]:
        assert (out / f).exists(), f
    idx = json.loads((out / "snapshots/index.json").read_text())
    assert idx["latest"] == 3 and idx["rounds"][0]["status"] == "pass"


def test_short_names():
    assert X.short_code("Red Bull Racing") == "RBR" and X.short_code("Kick Sauber") == "SAU" and X.short_code("Zed Motors") == "ZED"
    assert X._short("Haas F1 Team") == "Haas"
