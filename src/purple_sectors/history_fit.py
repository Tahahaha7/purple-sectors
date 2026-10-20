"""Fit the history model and turn it into 2026 driver priors; validate by predicting a held-out season.

Usage:
  python -m purple_sectors.history_fit --last 2025 --priors data/priors/drivers_2026.json   # production
  python -m purple_sectors.history_fit --last 2024 --holdout 2025 --report holdout.json      # validation
  python -m purple_sectors.history_fit --recovery --report recovery.json                     # recovery check
"""
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CUT = 1.07                      # drop best laps beyond 107% of the round-segment best (wet, aborted)


@dataclass
class History:
    obs: pd.DataFrame
    seasons: list
    teams: dict                 # season -> list of team ids (index = team number)
    drivers: list
    data: dict


def load(path: Path, first: int, last: int) -> pd.DataFrame:
    h = pd.read_csv(path)
    h = h[(h["season"] >= first) & (h["season"] <= last)]
    long = h.melt(id_vars=["season", "round", "driver", "team_id"], value_vars=["q1", "q2", "q3"], var_name="seg", value_name="t").dropna(subset=["t"])
    best = long.groupby(["season", "round", "seg"])["t"].transform("min")
    long = long[long["t"] <= best * CUT].copy()
    long["y"] = 100 * np.log(long["t"])
    return long.reset_index(drop=True)


def design(obs: pd.DataFrame) -> History:
    seasons = sorted(obs["season"].unique())
    teams = {s: sorted(obs.loc[obs["season"] == s, "team_id"].unique()) for s in seasons}
    T = max(len(v) for v in teams.values())
    drivers = sorted(obs["driver"].unique())
    rounds = {s: sorted(obs.loc[obs["season"] == s, "round"].unique()) for s in seasons}
    rnd_idx = {(s, r): i + 1 for s in seasons for i, r in enumerate(rounds[s])}
    bases = obs[["season", "round", "seg"]].drop_duplicates().reset_index(drop=True)
    base_idx = {tuple(v): i + 1 for i, v in enumerate(bases.to_numpy().tolist())}
    present = np.zeros((len(drivers), len(seasons)), dtype=int)
    for (d, s) in obs[["driver", "season"]].drop_duplicates().itertuples(index=False):
        present[drivers.index(d), seasons.index(s)] = 1
    y_ref = obs.groupby(["season", "round", "seg"])["y"].median().reindex(pd.MultiIndex.from_frame(bases)).to_numpy()
    data = dict(N=len(obs), S=len(seasons), T=T, RMAX=max(len(v) for v in rounds.values()),
                R=[len(rounds[s]) for s in seasons], D=len(drivers), A=len(bases),
                season=[seasons.index(s) + 1 for s in obs["season"]], rnd=[rnd_idx[(s, r)] for s, r in zip(obs["season"], obs["round"])],
                team=[teams[s].index(t) + 1 for s, t in zip(obs["season"], obs["team_id"])],
                driver=[drivers.index(d) + 1 for d in obs["driver"]],
                base=[base_idx[(s, r, g)] for s, r, g in zip(obs["season"], obs["round"], obs["seg"])],
                present=present.tolist(), y_ref=y_ref.tolist(), y=obs["y"].tolist())
    return History(obs, seasons, teams, drivers, data)


def priors(fit, H: History, next_season: int) -> dict:
    """Each driver's skill entering next_season (mean, sd in % of lap time), from their last season seen."""
    skill = fit.stan_variable("skill")            # draws x D x S
    drift = fit.stan_variable("sigma_drift")
    pres = np.array(H.data["present"])
    out = {}
    for i, d in enumerate(H.drivers):
        last = int(np.where(pres[i])[0].max())
        gap = (len(H.seasons) - 1 - last) + 1    # seasons from the last one raced to next_season
        draws = skill[:, i, last] + drift * np.sqrt(gap) * np.random.default_rng(i).normal(size=len(drift))
        out[d] = {"mean": round(float(draws.mean()), 4), "sd": round(float(draws.std()), 4),
                  "last_season": int(H.seasons[last]), "seasons": int(pres[i].sum())}
    return {"next_season": next_season, "units": "% of lap time vs the average driver; negative = faster",
            "sigma_driver": round(float(np.median(fit.stan_variable("sigma_driver"))), 4),
            "sigma_drift": round(float(np.median(drift)), 4), "drivers": out}


def season_gaps(obs: pd.DataFrame, season: int) -> pd.DataFrame:
    """Observed teammate gaps in a season: median over shared round-segments of y_a - y_b."""
    o = obs[obs["season"] == season]
    rows = []
    for team, g in o.groupby("team_id"):
        piv = g.pivot_table(index=["round", "seg"], columns="driver", values="y")
        ds = [d for d in piv.columns if piv[d].notna().sum() >= 5]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                both = piv[[ds[i], ds[j]]].dropna()
                if len(both) >= 5:
                    rows.append({"team": team, "a": ds[i], "b": ds[j], "gap": float((both[ds[i]] - both[ds[j]]).median()), "n": len(both)})
    return pd.DataFrame(rows)


def holdout(fit, H: History, full_obs: pd.DataFrame, season: int) -> dict:
    """Predict next-season teammate gaps from skill_next; compare with what happened."""
    nxt = fit.stan_variable("skill_next")        # draws x D
    obs = season_gaps(full_obs, season)
    rows = []
    for r in obs.itertuples():
        if r.a in H.drivers and r.b in H.drivers:
            g = nxt[:, H.drivers.index(r.a)] - nxt[:, H.drivers.index(r.b)]
            lo, med, hi = np.percentile(g, [5, 50, 95])
            rows.append({"pair": f"{r.a}-{r.b}", "team": r.team, "observed": round(r.gap, 3), "pred": round(med, 3),
                         "lo": round(lo, 3), "hi": round(hi, 3), "inside": bool(lo <= r.gap <= hi), "same_sign": bool(np.sign(med) == np.sign(r.gap))})
    df = pd.DataFrame(rows)
    return {"season": season, "pairs": rows, "coverage90": round(float(df["inside"].mean()), 3) if len(df) else None,
            "sign_right": round(float(df["same_sign"].mean()), 3) if len(df) else None,
            "corr": round(float(df["observed"].corr(df["pred"])), 3) if len(df) > 2 else None}


def simulate(H: History, rng, truth=None) -> tuple[np.ndarray, dict]:
    """Fake best laps on the real history design, from known parameters (recovery check)."""
    t = dict(sigma_car=0.9, sigma_dev=0.06, sigma_nu=0.3, sigma_driver=0.25, sigma_drift=0.06, sigma=0.25, nu=5.0)
    t.update(truth or {})
    d = H.data; S, T, RMAX, D = d["S"], d["T"], d["RMAX"], d["D"]
    car = np.zeros((S, T, RMAX))
    for s in range(S):
        w = t["sigma_car"] * rng.normal(size=T)
        for r in range(RMAX):
            if r: w = w + t["sigma_dev"] * rng.normal(size=T)
            c = w + t["sigma_nu"] * rng.normal(size=T); car[s, :, r] = c - c.mean()
    pres = np.array(d["present"]); walk = np.zeros((D, S)); walk[:, 0] = t["sigma_driver"] * rng.normal(size=D)
    for s in range(1, S): walk[:, s] = walk[:, s - 1] + t["sigma_drift"] * rng.normal(size=D)
    skill = walk - np.array([[walk[pres[:, s] == 1, s].mean() for s in range(S)]])
    i = lambda k: np.array(d[k]) - 1
    mu = np.array(d["y_ref"])[i("base")] + car[i("season"), i("team"), i("rnd")] + skill[i("driver"), i("season")]
    y = mu + t["sigma"] * rng.standard_t(t["nu"], size=len(mu))
    return y, {**t, "skill": skill}


def main(a) -> dict:
    from cmdstanpy import CmdStanModel
    obs = load(a.history, a.first, a.last)
    H = design(obs)
    data = dict(H.data); truth = None
    if a.recovery:
        y, truth = simulate(H, np.random.default_rng(a.seed)); data["y"] = y.tolist()
    fit = CmdStanModel(stan_file=str(a.stan)).sample(data=data, chains=4, parallel_chains=4, iter_warmup=a.warmup,
                                                     iter_sampling=a.samples, adapt_delta=0.95, inits=0.5, seed=a.seed, show_progress=False)
    div = int(fit.method_variables()["divergent__"].sum())
    rep = {"seasons": H.seasons, "N": H.data["N"], "drivers": len(H.drivers), "divergences": div,
           "medians": {k: round(float(np.median(fit.stan_variable(k))), 4) for k in ["sigma_car", "sigma_dev", "sigma_nu", "sigma_driver", "sigma_drift", "sigma", "nu"]}}
    if a.recovery:
        sk = fit.stan_variable("skill"); lo, hi = np.percentile(sk, [5, 95], axis=0); pres = np.array(H.data["present"]) == 1
        rep["recovery"] = {"skill_coverage90": round(float(((truth["skill"] >= lo) & (truth["skill"] <= hi))[pres].mean()), 3),
                           "truth": {k: v for k, v in truth.items() if k != "skill"}}
    if a.holdout:
        rep["holdout"] = holdout(fit, H, load(a.history, a.first, a.holdout), a.holdout)
    if a.priors:
        Path(a.priors).parent.mkdir(parents=True, exist_ok=True)
        Path(a.priors).write_text(json.dumps(priors(fit, H, a.last + 1), indent=1))
    print(json.dumps(rep, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        L = [f"## History fit {H.seasons[0]}–{H.seasons[-1]}" + (" (recovery)" if a.recovery else ""),
             f"{H.data['N']} best laps · {len(H.drivers)} drivers · divergences {div}",
             "", "| parameter | median |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in rep["medians"].items()]
        if "recovery" in rep:
            L += ["", f"**Recovery:** driver-season skill inside 90% interval: {rep['recovery']['skill_coverage90']:.0%}"]
        if "holdout" in rep:
            ho = rep["holdout"]
            L += ["", f"**Hold-out {ho['season']}:** predicted teammate gaps before seeing the season. "
                      f"Inside 90% interval: {ho['coverage90']:.0%} · right driver ahead: {ho['sign_right']:.0%} · correlation {ho['corr']}",
                  "", "| pair | team | predicted | 90% | observed |", "|---|---|---|---|---|"]
            L += [f"| {p['pair']} | {p['team']} | {p['pred']:+.2f} | [{p['lo']:+.2f}, {p['hi']:+.2f}] | {p['observed']:+.2f} {'✅' if p['inside'] else '❌'} |" for p in ho["pairs"]]
        Path(summary).open("a").write("\n".join(L) + "\n")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history", type=Path, default=Path("data/raw/history_quali_2022_2025.csv"))
    ap.add_argument("--stan", type=Path, default=Path("stan/history.stan"))
    ap.add_argument("--first", type=int, default=2022)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--holdout", type=int, default=None)
    ap.add_argument("--recovery", action="store_true")
    ap.add_argument("--priors", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    main(ap.parse_known_args()[0])
