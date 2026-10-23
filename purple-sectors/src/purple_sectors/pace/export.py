"""Posterior → site/data/pace.json, the one file the site reads.

  python -m purple_sectors.pace.export --posterior posterior_pace.npz --out site/data/pace.json

Everything is in % of lap time (negative = faster); the site multiplies by 0.9 to show
seconds on a 90-second lap. Intervals are [5%, 50%, 95%].
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import simulate as sim

Q = (5, 50, 95)
SEC_PER_PCT = 0.9
SHORT = {"mclaren": "MCL", "ferrari": "FER", "red bull": "RBR", "mercedes": "MER", "aston martin": "AMR",
         "alpine": "ALP", "williams": "WIL", "racing bulls": "RB", "rb": "RB", "alphatauri": "AT",
         "haas": "HAA", "sauber": "SAU", "audi": "AUD", "alfa romeo": "ALF", "cadillac": "CAD"}


def short(team: str) -> str:
    low = team.lower()
    for key, code in SHORT.items():
        if key in low:
            return code
    return "".join(ch for ch in team.upper() if ch.isalpha())[:3]


def q(x, nd=3) -> list:
    return [round(float(v), nd) for v in np.percentile(x, Q)]


def qi(x) -> list:
    return [int(round(float(v))) for v in np.percentile(x, Q)]


def load(path) -> tuple[dict, dict, dict]:
    npz = np.load(path)
    post = {k: npz[k] for k in npz.files}
    return post, json.loads(str(post["meta_json"])), json.loads(str(post["diag_json"]))


def build(post: dict, meta: dict, diag: dict, min_events: int | None = None, seed: int = 0) -> dict:
    seasons = meta["seasons"]
    s = len(seasons) - 1                                  # current season index
    teams = meta["teams"][str(seasons[s])]
    events = [e for e in meta["events"] if e["season"] == seasons[s]]
    r = len(events) - 1                                   # latest round index
    R = len(events)
    T = len(teams)
    drivers = meta["drivers"]
    pairs = [tuple(p) for p in meta["pairs"]]
    pair_of = {p: i for i, p in enumerate(pairs)}
    names = meta["names"]
    laps_season, laps_driver, laps_pair = post["season"] - 1, post["driver"] - 1, post["pair"] - 1

    # drivers shown: enough events in the current season; their seat = the car they drove most
    min_events = min_events or min(3, math.ceil(R / 3))
    shown, seat, seat_laps = [], {}, {}
    for d in range(len(drivers)):
        on = (laps_season == s) & (laps_driver == d)
        if on.sum() == 0:
            continue
        if len(np.unique(post["event"][on])) < min_events:
            continue
        counts = np.bincount(post["team"][on] - 1, minlength=T)
        shown.append(d)
        seat[d] = int(counts.argmax())
        seat_laps[d] = int(counts.max())
    n = len(shown)

    car_now = post["trend"][:, s, :T, r].astype(float)                   # draws x T: persistent pace now
    car_last = post["car"][:, s, :T, r].astype(float)                    # draws x T: what we saw at the last track
    car_trend = np.median(post["trend"][:, s, :T, :R], axis=0)            # T x R
    skill_now = post["skill"][:, shown, s].astype(float)                   # draws x n
    fit_seat = np.stack([post["fit"][:, pair_of[(d, s, seat[d])]] for d in shown], axis=1).astype(float)
    actual = car_now[:, [seat[d] for d in shown]] + skill_now + fit_seat   # draws x n: each driver in their seat
    sigma_fit = post["sigma_fit"].astype(float)
    rng = np.random.default_rng(seed)
    ndraw = car_now.shape[0]

    fastest = car_now.argmin(axis=1)
    cars = []
    for t_ in range(T):
        gap = car_now[:, t_] - car_now.min(axis=1)
        cars.append(dict(team=teams[t_], short=short(teams[t_]), pace=q(car_now[:, t_]), gap=q(gap),
                         last=q(car_last[:, t_] - car_last.min(axis=1)),
                         p_fastest=round(float((fastest == t_).mean()), 3),
                         trend=[round(float(v), 3) for v in car_trend[t_]],
                         drivers=[drivers[d] for d in shown if seat[d] == t_]))
    cars.sort(key=lambda c: c["pace"][1])
    car_order = [teams.index(c["team"]) for c in cars]

    ranks = skill_now.argsort(axis=1).argsort(axis=1) + 1
    team_name = {(si, ti): tm for si, sn in enumerate(seasons) for ti, tm in enumerate(meta["teams"][str(sn)])}
    teams_driven = {d: len({team_name[(p[1], p[2])] for p in pairs if p[0] == d}) for d in range(len(drivers))}
    rows = []
    for j, d in enumerate(shown):
        code = drivers[d]
        rows.append(dict(
            code=code, name=names[code]["name"], number=names[code]["number"],
            team=teams[seat[d]], short=short(teams[seat[d]]),
            skill=q(skill_now[:, j]), rank=qi(ranks[:, j]),
            p_fastest=round(float((ranks[:, j] == 1).mean()), 3), p_top3=round(float((ranks[:, j] <= 3).mean()), 3),
            fit=q(fit_seat[:, j]), teams_driven=teams_driven[d], moved=teams_driven[d] > 1,
            rookie=bool(meta["rookie"][d][s]), young=bool(meta["young"][d][s]), laps=int(seat_laps[d])))
    rows.sort(key=lambda x: x["skill"][1])
    driver_order = [shown.index(drivers.index(x["code"])) for x in rows]

    mates = []
    for t_ in range(T):
        here = [j for j, d in enumerate(shown) if seat[d] == t_]
        for i in range(len(here)):
            for k in range(i + 1, len(here)):
                a, b = here[i], here[k]
                gap = actual[:, a] - actual[:, b]
                if np.median(gap) > 0:
                    a, b, gap = b, a, -gap
                mates.append(dict(team=teams[t_], short=short(teams[t_]), a=drivers[shown[a]], b=drivers[shown[b]],
                                  gap=q(gap), p_a_ahead=round(float((gap < 0).mean()), 3),
                                  laps=int(seat_laps[shown[a]] + seat_laps[shown[b]])))
    mates.sort(key=lambda m: m["gap"][1])

    # counterfactual grid: every shown driver in every current car, rows and columns in display order
    mask = ~np.eye(n, dtype=bool)
    cells = []
    for j in driver_order:
        d = shown[j]
        row = []
        for t_ in car_order:
            key = (d, s, t_)
            seen = key in pair_of
            fit = post["fit"][:, pair_of[key]].astype(float) if seen else sigma_fit * rng.normal(size=ndraw)
            pace = car_now[:, t_] + skill_now[:, j] + fit
            pos = 1 + ((actual < pace[:, None]) & mask[j]).sum(axis=1)
            laps = int(((laps_pair == pair_of[key])).sum()) if seen else 0
            row.append(dict(pos=qi(pos), pace=q(pace), seen=seen, laps=laps, seat=t_ == seat[d]))
        cells.append(row)

    hypers = {k: q(post[k]) for k in sim.HYPERS + ["sigma_dev", "sigma_nu", "loss_bad"] if k in post}
    last = events[r]
    return dict(
        model="pace", generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        season=seasons[s], seasons=seasons, round=last["round"], n_rounds=R,
        event=last.get("name", ""), location=last.get("location", ""), date=last.get("date", ""),
        laps=int(len(post["y"])), laps_this_season=int((laps_season == s).sum()),
        units=dict(pace="% of lap time, negative = faster", sec_per_pct=SEC_PER_PCT, interval="5-95%"),
        status=diag.get("status", "unknown"), diagnostics=diag,
        execution=dict(p_bad=round(float(np.median(post["pi_bad"])), 3),
                       loss_clean_ms=int(round(float(np.median(post["loss_clean"])) * SEC_PER_PCT * 1000)),
                       loss_bad_ms=int(round(float(np.median(post["loss_bad"])) * SEC_PER_PCT * 1000))),
        hypers=hypers, connectivity=meta.get("connectivity", {}),
        cars=cars, drivers=rows, teammates=mates,
        grid=dict(drivers=[x["code"] for x in rows], cars=[c["short"] for c in cars],
                  teams=[c["team"] for c in cars], cells=cells))


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--posterior", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("site/data/pace.json"))
    ap.add_argument("--min-events", type=int, default=None, help="events a driver needs this season to be shown")
    a = ap.parse_known_args(argv)[0]
    post, meta, diag = load(a.posterior)
    out = build(post, meta, diag, a.min_events)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out} · {out['season']} round {out['round']} · {len(out['drivers'])} drivers · {len(out['cars'])} cars")
    return a.out


if __name__ == "__main__":
    main()
