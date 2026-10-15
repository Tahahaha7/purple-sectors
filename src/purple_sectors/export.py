"""Turn a fitted posterior (posterior_*.npz) into small JSON files for the website.

Usage:  python src/purple_sectors/export.py --posterior posterior_v12.npz \
            --laps data/raw/quali_laps_2026.csv --out site/data
"""
import argparse
import json
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


def soft(post, laps_csv: Path) -> dict:
    """Everything the SOFT (fan) page shows, in seconds on a 90-second lap."""
    raw = pd.read_csv(laps_csv)
    info = raw.groupby("driver").agg(name=("full_name", "first"), number=("driver_number", "first"))
    drivers, cars = [str(d) for d in post["drivers"]], [str(c) for c in post["cars"]]
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

    def person(i):
        code = drivers[i]
        return {"code": code, "name": str(info.loc[code, "name"]), "number": int(info.loc[code, "number"])}

    delta = post["delta"] * SEC
    d_idx, c_idx, e_idx = post["driver"], post["car"], post["event"]
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
                      "dots": np.round(np.quantile(gap, (np.arange(20) + 0.5) / 20), 3).tolist()})
    pairs.sort(key=lambda p: -p["p"])

    rng = np.random.default_rng(7)
    pick = rng.choice(now.shape[0], min(96, now.shape[0]), replace=False)
    meds = [r["med"] for r in car_rows]
    return {
        "model": "v1.2", "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "round": rounds[-1], "n_rounds": len(rounds), "event": str(last["event"].iloc[0]),
        "location": str(last["location"].iloc[0]), "date": str(last["date_start"].min())[:10].replace("-", "."),
        "laps": int(len(post["y"])),
        "car_share": [round(float(x), 3) for x in np.percentile(post["car_share"].mean(axis=1), Q)],
        "pi_bad": round(float(np.median(post["pi_bad"])), 3),
        "loss_clean_ms": int(round(float(np.median(post["loss_clean"])) * 900)),
        "car_spread": round(max(meds) - min(meds), 2),
        "mate_gap": round(float(np.median([abs(p["gap"]) for p in pairs])), 2),
        "cars": car_rows, "pairs": pairs,
        "draws": {_short(t): np.round(now[pick, c], 3).tolist() for c, t in enumerate(cars)},
    }


def main(posterior: Path, laps: Path, out: Path, model: str = "v1.2") -> Path:
    post = np.load(posterior, allow_pickle=True)
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
    ap.add_argument("--posterior", type=Path, required=True)
    ap.add_argument("--laps", type=Path, default=Path("data/raw/quali_laps_2026.csv"))
    ap.add_argument("--out", type=Path, default=Path("site/data"))
    ap.add_argument("--model", default="v1.2")
    args, _ = ap.parse_known_args()
    main(args.posterior, args.laps, args.out, args.model)
