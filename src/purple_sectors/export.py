"""Turn a fitted posterior (posterior_*.npz) into small JSON files for the website.

Usage:  python src/purple_sectors/export.py --posterior posterior_v12.npz \
            --laps data/raw/quali_laps_2026.csv --out site/data
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

Q = (5, 50, 95)  # 90% intervals


def _band(draws: np.ndarray) -> dict:
    lo, med, hi = np.percentile(draws, Q, axis=0)
    return {"median": np.round(med, 3).tolist(), "lo": np.round(lo, 3).tolist(),
            "hi": np.round(hi, 3).tolist()}


def events_meta(post, laps_csv: Path) -> list:
    raw = pd.read_csv(laps_csv)
    meta = raw.groupby("round").agg(name=("event", "first"), location=("location", "first"),
                                    date=("date_start", "min"))
    rounds = [int(r) for r in post["events"]]
    return [{"round": r, "name": meta.loc[r, "name"], "location": meta.loc[r, "location"],
             "date": str(meta.loc[r, "date"])[:10]} for r in rounds]


def car_pace(post) -> list:
    """Per team: development trend (theta), event pace (theta + nu), gap to fastest car,
    P(fastest car) per event, and change in trend from first to last event."""
    theta, nu = post["theta"], post["nu"]                 # draws x events x cars
    pace = theta + nu
    gap = pace - pace.min(axis=2, keepdims=True)
    fastest = pace.argmin(axis=2)
    change = theta[:, -1, :] - theta[:, 0, :]
    out = []
    for c, team in enumerate(post["cars"]):
        lo, med, hi = np.percentile(change[:, c], Q)
        out.append({
            "team": str(team),
            "trend": _band(theta[:, :, c]),
            "event_pace": _band(pace[:, :, c]),
            "gap_to_best": _band(gap[:, :, c]),
            "p_fastest": np.round((fastest == c).mean(axis=0), 3).tolist(),
            "season_change": {"median": round(float(med), 3), "lo": round(float(lo), 3),
                              "hi": round(float(hi), 3),
                              "p_improved": round(float((change[:, c] < 0).mean()), 3)},
        })
    order = np.argsort([np.mean(t["trend"]["median"]) for t in out])
    return [out[i] for i in order]


SEC = 0.9                       # % of lap time -> seconds on a typical 90-second lap
SHORT = {"Haas F1 Team": "Haas", "Red Bull Racing": "Red Bull"}


def _short(team: str) -> str:
    return SHORT.get(str(team), str(team))


QS = [0.025] + [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)] + [0.975]   # 21 quantiles


def _q(x) -> list:
    return np.round(np.quantile(np.asarray(x, dtype=float), QS), 3).tolist()


def _iv(x) -> list:
    return [round(float(v), 3) for v in np.percentile(np.asarray(x, dtype=float), Q)]


GRID = (-0.8, 0.8, 81)            # teammate-gap density grid, seconds on a 90 s lap


def density(draws: np.ndarray, grid=GRID) -> list:
    """Smoothed posterior density on a fixed grid (fine histogram + Gaussian kernel, no SciPy needed)."""
    lo, hi, n = grid
    draws = np.asarray(draws, dtype=float)
    x = np.linspace(lo, hi, n)
    fine = np.linspace(lo, hi, 8 * (n - 1) + 1)
    h, _ = np.histogram(draws, bins=np.r_[fine - (fine[1] - fine[0]) / 2, fine[-1] + (fine[1] - fine[0]) / 2])
    iqr = np.subtract(*np.percentile(draws, [75, 25]))
    bw = 0.9 * min(draws.std(), iqr / 1.34 if iqr > 0 else draws.std()) * len(draws) ** -0.2
    k = np.arange(-int(4 * bw / (fine[1] - fine[0])) - 1, int(4 * bw / (fine[1] - fine[0])) + 2) * (fine[1] - fine[0])
    smooth = np.convolve(h, np.exp(-0.5 * (k / bw) ** 2), mode="same")
    dens = smooth / (smooth.sum() * (fine[1] - fine[0])) * (np.isfinite(draws).mean())
    return np.round(np.interp(x, fine, dens), 2).tolist()


def teammate_pairs(post, info: pd.DataFrame) -> list:
    """For each car, the pair of drivers who shared it most, with the posterior gap (seconds)."""
    drivers, cars = [str(d) for d in post["drivers"]], [str(c) for c in post["cars"]]
    delta = post["delta"] * SEC
    d_idx, c_idx, e_idx = post["driver"], post["car"], post["event"]

    def person(i):
        code = drivers[i]
        return {"code": code, "name": str(info.loc[code, "name"]), "number": int(info.loc[code, "number"])}

    pairs = []
    for c, team in enumerate(cars):
        ds = sorted(set(d_idx[c_idx == c].tolist()))
        if len(ds) < 2:
            continue
        evs = {i: set(e_idx[(d_idx == i) & (c_idx == c)].tolist()) for i in ds}
        a, b = max(((x, y) for k, x in enumerate(ds) for y in ds[k + 1:]),
                   key=lambda p: len(evs[p[0]] & evs[p[1]]))
        gap = delta[:, a] - delta[:, b]
        if np.median(gap) > 0:
            a, b, gap = b, a, -gap
        lo, med, hi = np.percentile(gap, Q)
        pairs.append({"team": _short(team), "a": person(a), "b": person(b),
                      "gap": round(med, 3), "lo": round(lo, 3), "hi": round(hi, 3),
                      "p": round(float((gap < 0).mean()), 3), "shared": len(evs[a] & evs[b]),
                      "also": [{"code": drivers[i], "name": str(info.loc[drivers[i], "name"])}
                               for i in ds if i not in (a, b)],
                      "dots": np.round(np.quantile(gap, (np.arange(20) + 0.5) / 20), 3).tolist(),
                      "density": density(gap)})
    pairs.sort(key=lambda p: -p["p"])
    return pairs


PARAMS = ["sigma_car", "sigma_dev", "sigma_nu", "sigma_driver", "sigma", "pi_bad", "loss_clean",
          "loss_bad", "mu_gamma", "sigma_gamma", "beta_m", "beta_a"]


def snapshot(post, laps_csv: Path) -> dict:
    """Posterior after round k: one node per car, driver, event and global parameter (factor graph)."""
    raw = pd.read_csv(laps_csv)
    info = raw.groupby("driver").agg(name=("full_name", "first"), number=("driver_number", "first"))
    drivers, cars = [str(d) for d in post["drivers"]], [str(c) for c in post["cars"]]
    rounds = [int(r) for r in post["events"]]
    meta = raw.groupby("round").agg(name=("event", "first"), location=("location", "first"),
                                    date=("date_start", "min"))
    now = post["theta"][:, -1, :] * SEC
    fastest = now.argmin(axis=1)
    team_of = {d: cars[np.bincount(post["car"][post["driver"] == i]).argmax()] for i, d in enumerate(drivers)}
    laps_per = np.bincount(post["event"], minlength=len(rounds))
    tau = post["tau"]
    snap = {
        "round": rounds[-1], "name": str(meta.loc[rounds[-1], "name"]),
        "location": str(meta.loc[rounds[-1], "location"]),
        "date": str(meta.loc[rounds[-1], "date"])[:10].replace("-", "."),
        "laps": int(len(post["y"])), "quantiles": QS,
        "cars": [{"team": _short(t), "pace": _q(now[:, c]), "p_fast": round(float((fastest == c).mean()), 3)}
                 for c, t in enumerate(cars)],
        "drivers": [{"code": d, "name": str(info.loc[d, "name"]), "number": int(info.loc[d, "number"]),
                     "team": _short(team_of[d]), "skill": _q(skill_sec(post)[:, i])}
                    for i, d in enumerate(drivers)],
        "teammates": [{k: v for k, v in p.items() if k != "dots"} for p in teammate_pairs(post, info)],
        "grid": list(GRID),
        "events": [{"round": r, "name": str(meta.loc[r, "name"]), "laps": int(laps_per[k]),
                    "q1_q2": _iv(tau[:, k, 1] * SEC), "q2_q3": _iv((tau[:, k, 2] - tau[:, k, 1]) * SEC),
                    "gamma": _iv(post["gamma"][:, k] * SEC) if "gamma" in post.files else None}
                   for k, r in enumerate(rounds)],
        "params": {p: _iv(post[p]) for p in PARAMS if p in post.files},
        "car_share": _iv(post["car_share"].mean(axis=1)),
    }
    if "divergences" in post.files:
        snap["diagnostics"] = {"divergences": int(post["divergences"]), "draws": int(post["draws_total"]),
                               "max_rhat": round(float(post["max_rhat"]), 4), "passed": bool(post["passed"]),
                               "worst": str(post["worst"]) if "worst" in post.files else None,
                               "status": _status(post)}
    return snap


def _status(post) -> str:
    if "status" in post.files:
        return str(post["status"])
    return "pass" if "passed" not in post.files or bool(post["passed"]) else "fail"


DRAW_PARAMS = {"sigma_dev": SEC, "sigma_nu": SEC, "pi_bad": 1.0, "loss_clean": SEC, "sigma_driver": SEC, "sigma": SEC}


def draws_snapshot(post, n: int = 150) -> dict:
    """Joint posterior draws (same draw index across every quantity) for the Inter page's clouds."""
    S = post["theta"].shape[0]
    pick = np.linspace(0, S - 1, n).round().astype(int)
    now = post["theta"][pick, -1, :] * SEC
    rnd = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()
    return {
        "round": int(post["events"][-1]), "n": n,
        "cars": {_short(t): rnd(now[:, c]) for c, t in enumerate(post["cars"])},
        "drivers": {str(d): rnd(skill_sec(post)[pick, i]) for i, d in enumerate(post["drivers"])},
        "cross_team": is_cross_team(post),
        "params": {k: rnd(post[k][pick] * f) for k, f in DRAW_PARAMS.items() if k in post.files},
    }


def write_index(snap_dir: Path) -> Path:
    rows = []
    for f in sorted(p for p in snap_dir.glob("r*.json") if re.fullmatch(r"r\d+\.json", p.name)):
        s = json.loads(f.read_text())
        rows.append({"round": s["round"], "name": s["name"], "location": s["location"], "date": s["date"],
                     "file": f.name, "passed": s.get("diagnostics", {}).get("passed", True),
                     "status": s.get("diagnostics", {}).get("status")
                               or ("pass" if s.get("diagnostics", {}).get("passed", True) else "fail")})
    rows.sort(key=lambda r: r["round"])
    idx = snap_dir / "index.json"
    idx.write_text(json.dumps({"latest": rows[-1]["round"] if rows else None, "rounds": rows}, indent=1))
    print(f"wrote {idx} ({len(rows)} rounds)")
    return idx


def model_label(post) -> str:
    return "v2.0" if is_cross_team(post) else "v1.3"


def is_cross_team(post) -> bool:
    """v2.0+ posteriors: driver skill is comparable across teams (history priors)."""
    return "delta_prior_mean" in post.files


def skill_sec(post) -> np.ndarray:
    """Driver skill in seconds. Cross-team models: relative to the average driver on the grid (per draw)."""
    d = np.asarray(post["delta"], dtype=float) * SEC
    return d - d.mean(axis=1, keepdims=True) if is_cross_team(post) else d


def driver_ranking(post, info: pd.DataFrame) -> list:
    """Every driver ranked by skill with the car removed (v2.0+)."""
    drivers, cars = [str(d) for d in post["drivers"]], [str(c) for c in post["cars"]]
    sk = skill_sec(post); D = sk.shape[1]
    ranks = sk.argsort(axis=1).argsort(axis=1) + 1                       # 1 = fastest, per draw
    car_of = {i: cars[np.bincount(post["car"][post["driver"] == i]).argmax()] for i in range(D)}
    pm = np.asarray(post["delta_prior_mean"], dtype=float) * SEC
    kind = [str(k) for k in post["delta_prior_kind"]] if "delta_prior_kind" in post.files else ["history"] * D
    raw = np.asarray(post["delta"], dtype=float) * SEC
    rows = []
    for i, d in enumerate(drivers):
        lo, med, hi = np.percentile(sk[:, i], Q)
        rows.append({"code": d, "name": str(info.loc[d, "name"]), "number": int(info.loc[d, "number"]),
                     "team": _short(car_of[i]), "skill": [round(lo, 3), round(med, 3), round(hi, 3)],
                     "p_fastest": round(float((ranks[:, i] == 1).mean()), 3), "p_top3": round(float((ranks[:, i] <= 3).mean()), 3),
                     "rank": int(np.median(ranks[:, i])), "rank_lo": int(np.percentile(ranks[:, i], 10)), "rank_hi": int(np.percentile(ranks[:, i], 90)),
                     "history": kind[i] == "history", "vs_history": round(float(np.median(raw[:, i]) - pm[i]), 3)})
    rows.sort(key=lambda r: r["skill"][1])
    return rows


def soft(post, laps_csv: Path) -> dict:
    """Everything the SOFT (fan) page shows, in seconds on a 90-second lap."""
    raw = pd.read_csv(laps_csv)
    info = raw.groupby("driver").agg(name=("full_name", "first"), number=("driver_number", "first"))
    cars = [str(c) for c in post["cars"]]
    rounds = [int(r) for r in post["events"]]
    last = raw[raw["round"] == rounds[-1]]
    theta = post["theta"]
    now = theta[:, -1, :] * SEC                               # car pace at the latest round
    fastest = now.argmin(axis=1)
    change = (theta[:, -1, :] - theta[:, 0, :]) * SEC

    car_rows = []
    for c, team in enumerate(cars):
        lo, med, hi = np.percentile(now[:, c], Q)
        clo, cmed, chi = np.percentile(change[:, c], Q)
        car_rows.append({"team": _short(team), "lo": round(lo, 3), "med": round(med, 3), "hi": round(hi, 3),
                         "p_fast": round(float((fastest == c).mean()), 3),
                         "ch": round(cmed, 3), "ch_lo": round(clo, 3), "ch_hi": round(chi, 3),
                         "p_imp": round(float((change[:, c] < 0).mean()), 3)})
    car_rows.sort(key=lambda r: r["med"])

    pairs = [{k: v for k, v in p.items() if k != "density"} for p in teammate_pairs(post, info)]

    rng = np.random.default_rng(7)
    pick = rng.choice(now.shape[0], min(96, now.shape[0]), replace=False)
    meds = [r["med"] for r in car_rows]
    return {
        "model": model_label(post), "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "round": rounds[-1], "n_rounds": len(rounds), "event": str(last["event"].iloc[0]),
        "location": str(last["location"].iloc[0]), "date": str(last["date_start"].min())[:10].replace("-", "."),
        "laps": int(len(post["y"])),
        "car_share": [round(float(x), 3) for x in np.percentile(post["car_share"].mean(axis=1), Q)],
        "pi_bad": round(float(np.median(post["pi_bad"])), 3),
        "loss_clean_ms": int(round(float(np.median(post["loss_clean"])) * 900)),
        "car_spread": round(max(meds) - min(meds), 2),
        "mate_gap": round(float(np.median([abs(p["gap"]) for p in pairs])), 2),
        "status": _status(post), "cross_team": is_cross_team(post),
        "drivers": driver_ranking(post, info) if is_cross_team(post) else None,
        "cars": car_rows, "pairs": pairs,
        "draws": {_short(t): np.round(now[pick, c], 3).tolist() for c, t in enumerate(cars)},
    }


def main(posterior: Path, laps: Path, out: Path, model: str = "v1.3",
         snapshot_dir: Path | None = None, only_snapshot: bool = False) -> Path:
    post = np.load(posterior, allow_pickle=True)
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snap = snapshot(post, laps)
        sp = snapshot_dir / f"r{snap['round']:02d}.json"
        sp.write_text(json.dumps(snap, separators=(",", ":"), ensure_ascii=False))
        print(f"wrote {sp} ({sp.stat().st_size / 1024:.0f} KB)")
        dp = snapshot_dir / f"r{snap['round']:02d}_draws.json"
        dp.write_text(json.dumps(draws_snapshot(post), separators=(",", ":")))
        print(f"wrote {dp} ({dp.stat().st_size / 1024:.0f} KB)")
        write_index(snapshot_dir)
        from purple_sectors.forecast import write as write_forecast
        write_forecast(post, laps, snapshot_dir.parent)          # forecasts/r<next>.json + scorecard
        if only_snapshot:
            return sp
    doc = {
        "model": model,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "units": "% of lap time relative to the field average; negative = faster. "
                 "1% is about 900 ms on a 90 s lap.",
        "interval": "90%",
        "events": events_meta(post, laps),
        "cars": car_pace(post),
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "car_pace.json"
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    if "pi_bad" in post.files:                            # SOFT page needs the v1.2 mixture outputs
        sp = out / "soft.json"
        sp.write_text(json.dumps(soft(post, laps), separators=(",", ":"), ensure_ascii=False))
        print(f"wrote {sp} ({sp.stat().st_size / 1024:.0f} KB)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--posterior", type=Path)
    ap.add_argument("--laps", type=Path, default=Path("data/raw/quali_laps_2026.csv"))
    ap.add_argument("--out", type=Path, default=Path("site/data"))
    ap.add_argument("--model", default="v1.3")
    ap.add_argument("--snapshot-dir", type=Path, default=None, help="also write rXX.json + index.json here")
    ap.add_argument("--only-snapshot", action="store_true", help="skip car_pace.json / soft.json")
    ap.add_argument("--index-only", type=Path, default=None, help="just rebuild index.json in this folder")
    args, _ = ap.parse_known_args()
    if args.index_only:
        write_index(args.index_only)
        from purple_sectors.forecast import scorecard
        if (args.index_only.parent / "forecasts").exists():
            scorecard(args.index_only.parent / "forecasts", pd.read_csv(args.laps))
    else:
        main(args.posterior, args.laps, args.out, args.model, args.snapshot_dir, args.only_snapshot)
