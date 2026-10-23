"""Posterior → the JSON files the site reads.

  python -m purple_sectors.pace.export --posterior posterior_pace.npz --laps data/raw/quali_laps_2026.csv \
      --out site/data --snapshot-dir site/data/snapshots

Writes, for the latest season in the posterior:
  soft.json                      the Soft page (rankings, teammate battles, headline numbers)
  pace.json                      the counterfactual grid: every driver in every car
  snapshots/rXX.json, rXX_draws.json, index.json   one node per car, driver and event, for Medium and Inter
  forecasts/rXX.json, scorecard.json               next qualifying session and how past forecasts did

Everything is in % of lap time inside the posterior; the site shows seconds on a 90-second lap (1% = 0.9 s).
Intervals are 5-95%. Rank intervals are 10-90%, as the pages expect.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import simulate as sim

Q = (5, 50, 95)
SEC = 0.9
QS = [0.025] + [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)] + [0.975]   # 21 quantiles
GRID = (-0.8, 0.8, 81)            # teammate-gap density grid, seconds
SHORT = {"Haas F1 Team": "Haas", "Red Bull Racing": "Red Bull"}
CODE = {"mclaren": "MCL", "ferrari": "FER", "red bull": "RBR", "mercedes": "MER", "aston martin": "AMR",
        "alpine": "ALP", "williams": "WIL", "racing bulls": "RB", "rb": "RB", "alphatauri": "AT",
        "haas": "HAA", "sauber": "SAU", "audi": "AUD", "alfa romeo": "ALF", "cadillac": "CAD"}
SEGS = ["Q1", "Q2", "Q3"]


def _short(team: str) -> str:
    return SHORT.get(str(team), str(team))


def short_code(team: str) -> str:
    low = team.lower()
    for key, code in CODE.items():
        if key in low:
            return code
    return "".join(ch for ch in team.upper() if ch.isalpha())[:3]


def _q(x) -> list:
    return np.round(np.quantile(np.asarray(x, dtype=float), QS), 3).tolist()


def _iv(x) -> list:
    return [round(float(v), 3) for v in np.percentile(np.asarray(x, dtype=float), Q)]


def _qi(x) -> list:
    return [int(round(float(v))) for v in np.percentile(x, Q)]


def density(draws: np.ndarray, grid=GRID) -> list:
    """Smoothed posterior density on a fixed grid (fine histogram + Gaussian kernel)."""
    lo, hi, n = grid
    draws = np.asarray(draws, dtype=float)
    x = np.linspace(lo, hi, n)
    fine = np.linspace(lo, hi, 8 * (n - 1) + 1)
    h, _ = np.histogram(draws, bins=np.r_[fine - (fine[1] - fine[0]) / 2, fine[-1] + (fine[1] - fine[0]) / 2])
    iqr = np.subtract(*np.percentile(draws, [75, 25]))
    bw = 0.9 * min(draws.std(), iqr / 1.34 if iqr > 0 else draws.std()) * len(draws) ** -0.2
    bw = max(bw, 1e-3)
    k = np.arange(-int(4 * bw / (fine[1] - fine[0])) - 1, int(4 * bw / (fine[1] - fine[0])) + 2) * (fine[1] - fine[0])
    smooth = np.convolve(h, np.exp(-0.5 * (k / bw) ** 2), mode="same")
    dens = smooth / max(smooth.sum() * (fine[1] - fine[0]), 1e-12) * (np.isfinite(draws).mean())
    return np.round(np.interp(x, fine, dens), 2).tolist()


# ----------------------------------------------------------------------------- the season view

def load(path) -> tuple[dict, dict, dict]:
    npz = np.load(path)
    post = {k: npz[k] for k in npz.files}
    return post, json.loads(str(post["meta_json"])), json.loads(str(post["diag_json"]))


def season_view(post: dict, meta: dict, diag: dict, min_events: int | None = None) -> dict:
    """Everything about the latest season, in the layout the page builders below expect.

    Drivers shown: those with enough events this season (3, or a third of the rounds early on).
    Their seat is the car they drove most. Skill is relative to the season's grid; car pace is
    the persistent part (`theta`, the trend) plus a one-off per round (`nu`).
    """
    seasons = meta["seasons"]
    s = len(seasons) - 1
    teams = meta["teams"][str(seasons[s])]
    T = len(teams)
    events = [e for e in meta["events"] if e["season"] == seasons[s]]
    R = len(events)
    rounds = [int(e["round"]) for e in events]
    lap_s = post["season"] - 1
    on = lap_s == s
    lap_driver, lap_team, lap_rnd = post["driver"][on] - 1, post["team"][on] - 1, post["rnd"][on] - 1
    lap_base, lap_y = post["base"][on] - 1, post["y"][on]
    lap_pbad = post["p_bad_mean"][on] if "p_bad_mean" in post else np.zeros(int(on.sum()))
    bases = [tuple(b) for b in meta["bases"]]
    seg_of_base = np.array([SEGS.index(b[2]) for b in bases])
    lap_seg = seg_of_base[lap_base]

    min_events = min_events or min(3, math.ceil(R / 3))
    all_drivers = meta["drivers"]
    didx, seat, n_laps = [], {}, {}
    for d in range(len(all_drivers)):
        mine = lap_driver == d
        if mine.sum() == 0 or len(np.unique(lap_rnd[mine])) < min_events:
            continue
        counts = np.bincount(lap_team[mine], minlength=T)
        didx.append(d)
        seat[d] = int(counts.argmax())
        n_laps[d] = int(mine.sum())
    drivers = [all_drivers[d] for d in didx]
    local = {d: j for j, d in enumerate(didx)}

    pairs = [tuple(p) for p in meta["pairs"]]
    pair_of = {p: i for i, p in enumerate(pairs)}
    theta = np.transpose(post["trend"][:, s, :T, :R].astype(float), (0, 2, 1))     # draws x R x T
    pace = np.transpose(post["car"][:, s, :T, :R].astype(float), (0, 2, 1))
    delta = post["skill"][:, didx, s].astype(float)
    fit_seat = np.stack([post["fit"][:, pair_of[(d, s, seat[d])]] for d in didx], axis=1).astype(float) \
        if didx else np.zeros((delta.shape[0], 0))
    prev = np.full_like(delta, np.nan)
    has_prev = np.zeros(len(didx), dtype=bool)
    if s > 0:
        present = np.array(meta["present"])
        for j, d in enumerate(didx):
            if present[d, :s].any():
                last = int(np.where(present[d, :s])[0].max())
                prev[:, j] = post["skill"][:, d, last]
                has_prev[j] = True

    base_of = {b: i for i, b in enumerate(bases)}
    alpha = post["alpha"].astype(float)
    tau = np.zeros((alpha.shape[0], R, 3))
    for k, e in enumerate(events):
        q1 = base_of.get((seasons[s], e["round"], "Q1"))
        for g, seg in enumerate(SEGS[1:], start=1):
            b = base_of.get((seasons[s], e["round"], seg))
            if q1 is not None and b is not None:
                tau[:, k, g] = alpha[:, b] - alpha[:, q1]
    event_ids = [e["e"] - 1 for e in events]
    gamma = post["gamma"][:, event_ids].astype(float)

    H = {k: post[k].astype(float) for k in sim.HYPERS + ["sigma_dev", "sigma_nu", "loss_bad"] if k in post}
    H["beta_m"], H["beta_a"], H["rho_dev"], H["sigma_car_event"] = H["beta_medium"], H["beta_age"], H["rho"], H["sigma_car_round"]
    H["m"] = np.stack([tau[:, :, 1].mean(axis=1), (tau[:, :, 2] - tau[:, :, 1]).mean(axis=1)], axis=1)

    vc = pace.var(axis=2, ddof=1)                                                  # draws x R
    vd = delta.var(axis=1, ddof=1)[:, None] if delta.shape[1] > 1 else np.zeros((delta.shape[0], 1))
    car_share = vc / np.maximum(vc + vd, 1e-12)

    fits = {}
    for d in didx:
        for t in range(T):
            key = (d, s, t)
            if key in pair_of:
                fits.setdefault(all_drivers[d], {})[teams[t]] = post["fit"][:, pair_of[key]].astype(float)

    keep = np.array([lap in local for lap in lap_driver])
    names = meta.get("names", {})
    return dict(
        s=s, season=int(seasons[s]), seasons=seasons, teams=teams, T=T, R=R, events=events, rounds=rounds,
        drivers=drivers, didx=didx, seat={all_drivers[d]: teams[seat[d]] for d in didx}, seat_idx=[seat[d] for d in didx],
        n_laps=[n_laps[d] for d in didx], laps_season=int(on.sum()),
        names={d: names.get(d, dict(name=d, number=0)) for d in drivers},
        theta=theta, nu=pace - theta, pace=pace, delta=delta, fit_seat=fit_seat, prev=prev, has_prev=has_prev,
        tau=tau, gamma=gamma, hypers=H, car_share=car_share, sigma_fit=post["sigma_fit"].astype(float), fits=fits,
        lap_driver=np.array([local[d] for d in lap_driver[keep]]), lap_car=lap_team[keep], lap_event=lap_rnd[keep],
        lap_seg=lap_seg[keep], lap_y=lap_y[keep], lap_pbad=lap_pbad[keep],
        rookie=[bool(meta["rookie"][d][s]) for d in didx], young=[bool(meta["young"][d][s]) for d in didx],
        teams_driven=[len({meta["teams"][str(seasons[p[1]])][p[2]] for p in pairs if p[0] == d}) for d in didx],
        diag=diag)


# ----------------------------------------------------------------------------- builders

def person(V, j) -> dict:
    code = V["drivers"][j]
    return {"code": code, "name": str(V["names"][code]["name"]), "number": int(V["names"][code]["number"])}


def teammate_pairs(V) -> list:
    """Per car, the two drivers who shared it most this season, with the posterior gap in this car (seconds)."""
    skill = (V["delta"] + V["fit_seat"]) * SEC
    pairs = []
    for t, team in enumerate(V["teams"]):
        js = sorted(set(V["lap_driver"][V["lap_car"] == t].tolist()))
        js = [j for j in js if V["seat_idx"][j] == t]
        if len(js) < 2:
            continue
        evs = {j: set(V["lap_event"][(V["lap_driver"] == j) & (V["lap_car"] == t)].tolist()) for j in js}
        a, b = max(((x, y) for k, x in enumerate(js) for y in js[k + 1:]), key=lambda p: len(evs[p[0]] & evs[p[1]]))
        gap = skill[:, a] - skill[:, b]
        if np.median(gap) > 0:
            a, b, gap = b, a, -gap
        lo, med, hi = np.percentile(gap, Q)
        pairs.append({"team": _short(team), "a": person(V, a), "b": person(V, b),
                      "gap": round(float(med), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                      "p": round(float((gap < 0).mean()), 3), "shared": len(evs[a] & evs[b]),
                      "also": [person(V, j) for j in js if j not in (a, b)],
                      "dots": np.round(np.quantile(gap, (np.arange(20) + 0.5) / 20), 3).tolist(),
                      "density": density(gap)})
    pairs.sort(key=lambda p: -p["p"])
    return pairs


def driver_ranking(V) -> list:
    """Every shown driver ranked by skill with the car removed."""
    sk = V["delta"] * SEC
    ranks = sk.argsort(axis=1).argsort(axis=1) + 1
    rows = []
    for j, code in enumerate(V["drivers"]):
        lo, med, hi = np.percentile(sk[:, j], Q)
        prev = V["prev"][:, j]
        moved = float(np.median(sk[:, j] - prev * SEC)) if V["has_prev"][j] else 0.0
        rows.append({**person(V, j), "team": _short(V["seat"][code]),
                     "skill": [round(float(lo), 3), round(float(med), 3), round(float(hi), 3)],
                     "p_fastest": round(float((ranks[:, j] == 1).mean()), 3), "p_top3": round(float((ranks[:, j] <= 3).mean()), 3),
                     "rank": int(np.median(ranks[:, j])), "rank_lo": int(np.percentile(ranks[:, j], 10)),
                     "rank_hi": int(np.percentile(ranks[:, j], 90)),
                     "history": bool(V["has_prev"][j]), "vs_history": round(moved, 3),
                     "fit": _iv(V["fit_seat"][:, j] * SEC), "moved_teams": V["teams_driven"][j] > 1,
                     "rookie": V["rookie"][j], "laps": V["n_laps"][j]})
    rows.sort(key=lambda r: r["skill"][1])
    return rows


def soft(V) -> dict:
    """Everything the Soft page shows, in seconds on a 90-second lap."""
    theta = V["theta"]
    now = theta[:, -1, :] * SEC
    fastest = now.argmin(axis=1)
    change = (theta[:, -1, :] - theta[:, 0, :]) * SEC
    car_rows = []
    for c, team in enumerate(V["teams"]):
        lo, med, hi = np.percentile(now[:, c], Q)
        clo, cmed, chi = np.percentile(change[:, c], Q)
        car_rows.append({"team": _short(team), "lo": round(float(lo), 3), "med": round(float(med), 3), "hi": round(float(hi), 3),
                         "p_fast": round(float((fastest == c).mean()), 3),
                         "ch": round(float(cmed), 3), "ch_lo": round(float(clo), 3), "ch_hi": round(float(chi), 3),
                         "p_imp": round(float((change[:, c] < 0).mean()), 3)})
    car_rows.sort(key=lambda r: r["med"])
    pairs = [{k: v for k, v in p.items() if k != "density"} for p in teammate_pairs(V)]
    rng = np.random.default_rng(7)
    pick = rng.choice(now.shape[0], min(96, now.shape[0]), replace=False)
    meds = [r["med"] for r in car_rows]
    last = V["events"][-1]
    return {
        "model": "pace", "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": V["season"], "round": V["rounds"][-1], "n_rounds": V["R"], "event": last.get("name", ""),
        "location": last.get("location", ""), "date": str(last.get("date", ""))[:10].replace("-", "."),
        "laps": V["laps_season"],
        "car_share": _iv(V["car_share"].mean(axis=1)),
        "pi_bad": round(float(np.median(V["hypers"]["pi_bad"])), 3),
        "loss_clean_ms": int(round(float(np.median(V["hypers"]["loss_clean"])) * 900)),
        "car_spread": round(max(meds) - min(meds), 2) if meds else 0.0,
        "mate_gap": round(float(np.median([abs(p["gap"]) for p in pairs])), 2) if pairs else 0.0,
        "status": V["diag"].get("status", "pass"), "cross_team": True,
        "drivers": driver_ranking(V), "cars": car_rows, "pairs": pairs,
        "draws": {_short(t): np.round(now[pick, c], 3).tolist() for c, t in enumerate(V["teams"])},
    }


PARAMS = ["sigma_car", "sigma_dev", "sigma_nu", "sigma_driver", "sigma", "pi_bad", "loss_clean", "loss_bad",
          "mu_gamma", "sigma_gamma", "beta_m", "beta_a", "rho_dev", "sigma_fit", "sigma_drift", "sigma_young",
          "mu_rookie", "mu_young"]


def snapshot(V) -> dict:
    """Posterior after the latest round: one node per car, driver, event and shared parameter."""
    now = V["theta"][:, -1, :] * SEC
    fastest = now.argmin(axis=1)
    laps_per = np.bincount(V["lap_event"], minlength=V["R"])
    last = V["events"][-1]
    d = V["diag"]
    return {
        "round": V["rounds"][-1], "season": V["season"], "name": last.get("name", ""), "location": last.get("location", ""),
        "date": str(last.get("date", ""))[:10].replace("-", "."), "laps": V["laps_season"], "quantiles": QS,
        "cars": [{"team": _short(t), "pace": _q(now[:, c]), "p_fast": round(float((fastest == c).mean()), 3)}
                 for c, t in enumerate(V["teams"])],
        "drivers": [{**person(V, j), "team": _short(V["seat"][code]), "skill": _q(V["delta"][:, j] * SEC),
                     "history": bool(V["has_prev"][j])} for j, code in enumerate(V["drivers"])],
        "teammates": [{k: v for k, v in p.items() if k != "dots"} for p in teammate_pairs(V)],
        "grid": list(GRID),
        "events": [{"round": r, "name": e.get("name", ""), "laps": int(laps_per[k]),
                    "q1_q2": _iv(V["tau"][:, k, 1] * SEC), "q2_q3": _iv((V["tau"][:, k, 2] - V["tau"][:, k, 1]) * SEC),
                    "gamma": _iv(V["gamma"][:, k] * SEC)} for k, (r, e) in enumerate(zip(V["rounds"], V["events"]))],
        "params": {p: _iv(V["hypers"][p]) for p in PARAMS if p in V["hypers"]},
        "car_share": _iv(V["car_share"].mean(axis=1)),
        "diagnostics": {"divergences": int(d.get("divergences", 0)), "draws": int(d.get("draws", 0)),
                        "max_rhat": round(float(d.get("max_rhat", 1.0)), 4), "passed": d.get("status", "pass") != "fail",
                        "worst": d.get("worst"), "status": d.get("status", "pass"),
                        "min_ess_bulk": round(float(d.get("min_ess_bulk", 0))), "min_ess_tail": round(float(d.get("min_ess_tail", 0)))},
    }


DRAW_PARAMS = {"sigma_dev": SEC, "sigma_nu": SEC, "pi_bad": 1.0, "loss_clean": SEC, "sigma_driver": SEC,
               "sigma": SEC, "sigma_fit": SEC}


def draws_snapshot(V, n: int = 150) -> dict:
    """Joint posterior draws (same draw index everywhere) for the Inter page's clouds and counterfactuals."""
    S = V["theta"].shape[0]
    pick = np.linspace(0, S - 1, min(n, S)).round().astype(int)
    now = V["theta"][pick, -1, :] * SEC
    rnd = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()
    return {
        "round": V["rounds"][-1], "n": int(len(pick)),
        "cars": {_short(t): rnd(now[:, c]) for c, t in enumerate(V["teams"])},
        "drivers": {code: rnd(V["delta"][pick, j] * SEC) for j, code in enumerate(V["drivers"])},
        "fit": {code: {_short(team): rnd(draws[pick] * SEC) for team, draws in by_team.items()}
                for code, by_team in V["fits"].items() if code in V["drivers"]},
        "sigma_fit": rnd(V["sigma_fit"][pick] * SEC),
        "cross_team": True,
        "params": {k: rnd(V["hypers"][k][pick] * f) for k, f in DRAW_PARAMS.items() if k in V["hypers"]},
    }


def grid(V, seed: int = 0) -> dict:
    """Every shown driver in every current car, against the rest of the grid in their actual seats."""
    n = len(V["drivers"])
    car_now = V["theta"][:, -1, :]
    actual = car_now[:, V["seat_idx"]] + V["delta"] + V["fit_seat"]                   # draws x n
    rng = np.random.default_rng(seed)
    ndraw = car_now.shape[0]
    car_order = [int(t) for t in np.argsort(np.median(car_now, axis=0))]
    driver_order = [int(j) for j in np.argsort(np.median(V["delta"], axis=0))]
    mask = ~np.eye(n, dtype=bool)
    cells = []
    for j in driver_order:
        code = V["drivers"][j]
        row = []
        for t in car_order:
            team = V["teams"][t]
            seen = team in V["fits"].get(code, {})
            fit = V["fits"][code][team] if seen else V["sigma_fit"] * rng.normal(size=ndraw)
            pc = car_now[:, t] + V["delta"][:, j] + fit
            pos = 1 + ((actual < pc[:, None]) & mask[j]).sum(axis=1)
            row.append({"pos": _qi(pos), "pace": _iv(pc * SEC), "p_first": round(float((pos == 1).mean()), 3),
                        "seen": bool(seen), "seat": bool(V["seat_idx"][j] == t)})
        cells.append(row)
    last = V["events"][-1]
    return {
        "model": "pace", "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": V["season"], "round": V["rounds"][-1], "event": last.get("name", ""), "date": str(last.get("date", ""))[:10],
        "units": {"pace": "seconds on a 90-second lap, negative = faster", "interval": "5-95%"},
        "drivers": [{**person(V, j), "team": _short(V["seat"][V["drivers"][j]]), "short": short_code(V["seat"][V["drivers"][j]]),
                     "skill": _iv(V["delta"][:, j] * SEC), "moved": V["teams_driven"][j] > 1} for j in driver_order],
        "grid": {"drivers": [V["drivers"][j] for j in driver_order], "cars": [short_code(V["teams"][t]) for t in car_order],
                 "teams": [_short(V["teams"][t]) for t in car_order], "cells": cells},
    }


def write_index(snap_dir: Path) -> Path:
    rows = []
    for f in sorted(p for p in Path(snap_dir).glob("r*.json") if re.fullmatch(r"r\d+\.json", p.name)):
        s = json.loads(f.read_text())
        dg = s.get("diagnostics", {})
        rows.append({"round": s["round"], "name": s["name"], "location": s["location"], "date": s["date"], "file": f.name,
                     "passed": dg.get("passed", True), "status": dg.get("status") or ("pass" if dg.get("passed", True) else "fail")})
    rows.sort(key=lambda r: r["round"])
    idx = Path(snap_dir) / "index.json"
    idx.write_text(json.dumps({"latest": rows[-1]["round"] if rows else None, "rounds": rows}, indent=1))
    print(f"wrote {idx} ({len(rows)} rounds)")
    return idx


def _write(path: Path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def main(posterior: Path, laps: Path, out: Path, snapshot_dir: Path | None = None, only_snapshot: bool = False) -> Path:
    post, meta, diag = load(posterior)
    V = season_view(post, meta, diag)
    from .forecast import write as write_forecast
    if snapshot_dir is not None:
        snap = snapshot(V)
        _write(Path(snapshot_dir) / f"r{snap['round']:02d}.json", snap)
        _write(Path(snapshot_dir) / f"r{snap['round']:02d}_draws.json", draws_snapshot(V))
        write_index(snapshot_dir)
        write_forecast(V, laps, Path(snapshot_dir).parent)
        if only_snapshot:
            return Path(snapshot_dir)
    _write(Path(out) / "soft.json", soft(V))
    _write(Path(out) / "pace.json", grid(V))
    return Path(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--posterior", type=Path)
    ap.add_argument("--laps", type=Path, default=Path("data/raw/quali_laps_2026.csv"), help="this season's lap file (schedule + scoring)")
    ap.add_argument("--out", type=Path, default=Path("site/data"))
    ap.add_argument("--snapshot-dir", type=Path, default=None, help="also write rXX.json, rXX_draws.json, index.json and forecasts here")
    ap.add_argument("--only-snapshot", action="store_true", help="skip soft.json and pace.json (backfill)")
    ap.add_argument("--index-only", type=Path, default=None, help="just rebuild index.json (and the scorecard) in this folder")
    args, _ = ap.parse_known_args()
    if args.index_only:
        write_index(args.index_only)
        from .forecast import scorecard
        if (args.index_only.parent / "forecasts").exists():
            scorecard(args.index_only.parent / "forecasts", pd.read_csv(args.laps))
    else:
        main(args.posterior, args.laps, args.out, args.snapshot_dir, args.only_snapshot)
