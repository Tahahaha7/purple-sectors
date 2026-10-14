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
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--posterior", type=Path, required=True)
    ap.add_argument("--laps", type=Path, default=Path("data/raw/quali_laps_2026.csv"))
    ap.add_argument("--out", type=Path, default=Path("site/data"))
    ap.add_argument("--model", default="v1.2")
    args, _ = ap.parse_known_args()
    main(args.posterior, args.laps, args.out, args.model)
