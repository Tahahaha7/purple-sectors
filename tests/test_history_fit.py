import numpy as np
import pandas as pd

import purple_sectors.history_fit as HF

LINEUPS = {2024: {"red": ["AAA", "BBB"], "blue": ["CCC", "DDD"], "green": ["EEE", "FFF"]},
           2025: {"red": ["AAA", "CCC"], "blue": ["BBB", "DDD"], "green": ["EEE", "GGG"]}}   # B and C swap; G is new


def _history(tmp_path, rounds=6, slow=None):
    rng = np.random.default_rng(0); rows = []
    for s, teams in LINEUPS.items():
        for r in range(1, rounds + 1):
            for k, (team, ds) in enumerate(teams.items()):
                for j, d in enumerate(ds):
                    base = 90 + 0.4 * k + 0.1 * j + rng.normal(0, 0.05)
                    rows.append(dict(season=s, round=r, driver=d, team_id=team, team=team.title(), name=d,
                                     q1=base + 0.4, q2=base + 0.2 if k < 2 else None, q3=base if k < 1 else None))
    h = pd.DataFrame(rows)
    if slow:
        h.loc[0, "q1"] = h.loc[0, "q1"] * slow
    p = tmp_path / "hist.csv"; h.to_csv(p, index=False)
    return p


def test_load_cuts_slow_laps_and_design_indexes(tmp_path):
    obs = HF.load(_history(tmp_path, slow=1.2), 2024, 2025)
    assert obs["t"].max() < 95                                            # the 120% lap is gone
    H = HF.design(obs)
    d = H.data
    assert d["S"] == 2 and d["T"] == 3 and d["D"] == 7 and d["N"] == len(obs)
    assert min(d["team"]) == 1 and max(d["team"]) == 3 and max(d["rnd"]) <= d["RMAX"]
    pres = np.array(d["present"])
    assert pres[H.drivers.index("GGG")].tolist() == [0, 1] and pres[H.drivers.index("FFF")].tolist() == [1, 0]


def test_simulated_history_is_well_formed(tmp_path):
    H = HF.design(HF.load(_history(tmp_path), 2024, 2025))
    y, truth = HF.simulate(H, np.random.default_rng(1))
    assert y.shape == (H.data["N"],) and np.isfinite(y).all()
    pres = np.array(H.data["present"]) == 1
    assert np.allclose([truth["skill"][pres[:, s], s].mean() for s in range(2)], 0, atol=1e-9)   # centred per season


class _Fit:
    def __init__(self, D, S, n=500):
        rng = np.random.default_rng(2)
        self.v = {"skill": rng.normal(np.linspace(-0.3, 0.3, D)[None, :, None], 0.05, (n, D, S)),
                  "sigma_drift": np.abs(rng.normal(0.05, 0.01, n)), "sigma_driver": np.abs(rng.normal(0.25, 0.02, n))}
        self.v["skill_next"] = self.v["skill"][:, :, -1]

    def stan_variable(self, k):
        return self.v[k]


def test_priors_widen_for_drivers_who_missed_seasons(tmp_path):
    H = HF.design(HF.load(_history(tmp_path), 2024, 2025))
    pr = HF.priors(_Fit(H.data["D"], 2), H, 2026)
    assert pr["next_season"] == 2026 and set(pr["drivers"]) == set(H.drivers)
    assert pr["drivers"]["FFF"]["last_season"] == 2024 and pr["drivers"]["FFF"]["sd"] > pr["drivers"]["EEE"]["sd"]


def test_holdout_compares_predicted_and_observed_gaps(tmp_path):
    path = _history(tmp_path)
    H = HF.design(HF.load(path, 2024, 2024))
    out = HF.holdout(_Fit(H.data["D"], 1), H, HF.load(path, 2024, 2025), 2025)
    pairs = {p["pair"] for p in out["pairs"]}
    assert "AAA-CCC" in pairs and "BBB-DDD" in pairs                      # new 2025 pairings, predicted from 2024
    assert "EEE-GGG" not in pairs                                         # GGG has no history: not predictable
    assert out["coverage90"] is not None


def test_main_writes_json_reports_with_numpy_values(tmp_path):
    import json, sys, types
    path = _history(tmp_path)

    class Fit(_Fit):
        chains, num_draws_sampling = 4, 125
        def __init__(self, D, S):
            super().__init__(D, S)
            rng = np.random.default_rng(3)
            for k in ["sigma_car", "sigma_dev", "sigma_nu", "sigma", "nu"]:
                self.v[k] = np.abs(rng.normal(1, 0.1, 500))
        def method_variables(self):
            return {"divergent__": np.zeros((125, 4))}
        def draws_pd(self, vars):
            cols = {}
            for k in vars:
                a = self.v[k].reshape(500, -1)
                for i in range(a.shape[1]):
                    cols[f"{k}[{i}]"] = a[:, i]
            df = pd.DataFrame(cols); df.insert(0, "chain__", np.repeat(np.arange(1, 5), 125)); return df

    class Model:
        def __init__(self, stan_file): pass
        def sample(self, data, **kw): return Fit(data["D"], data["S"])
    cs = sys.modules["cmdstanpy"]; original = cs.CmdStanModel
    cs.CmdStanModel = Model
    args = types.SimpleNamespace(history=path, stan=None, first=2024, last=2025, holdout=None, recovery=False,
                                 priors=tmp_path / "p.json", report=tmp_path / "r.json", warmup=10, samples=10, seed=1)
    try:
        rep = HF.main(args)
    finally:
        cs.CmdStanModel = original
    assert json.loads((tmp_path / "r.json").read_text())["seasons"] == [2024, 2025]
    assert "AAA" in json.loads((tmp_path / "p.json").read_text())["drivers"]
    assert "max_rhat" in rep
