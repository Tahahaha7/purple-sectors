"""Next-qualifying forecast and its scorecard.

From a fitted posterior, simulate the next qualifying session once per draw: car pace moves forward
to the next event, each car gets a fresh track fit, every driver sets a few laps per segment, and the
knockout format (Q1, Q2, Q3) is played out. Then, once the real session has happened, score it.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEC = 0.9


def _short(team: str) -> str:
    return {"Haas F1 Team": "Haas", "Red Bull Racing": "Red Bull"}.get(str(team), str(team))


def field(post) -> list[tuple[int, int]]:
    """(driver index, car index) for the expected grid: per car, the two most recent drivers."""
    d_idx, c_idx, e_idx = post["driver"], post["car"], post["event"]
    out = []
    for c in range(len(post["cars"])):
        ds = sorted(set(d_idx[c_idx == c].tolist()), key=lambda d: -e_idx[(d_idx == d) & (c_idx == c)].max())
        out += [(d, c) for d in ds[:2]]
    return out


def simulate(post, weeks: float, laps_per_seg: int = 2, seed: int = 2026, n: int | None = None) -> tuple[list, np.ndarray]:
    """Grid positions (draws x drivers) for the next qualifying session."""
    rng = np.random.default_rng(seed)
    S = post["theta"].shape[0] if n is None else min(n, post["theta"].shape[0])
    pick = np.linspace(0, post["theta"].shape[0] - 1, S).round().astype(int)
    g = lambda k: np.asarray(post[k], dtype=float)[pick]
    F = field(post); D = len(F); cars = np.array([c for _, c in F]); drv = np.array([d for d, _ in F])
    C = post["theta"].shape[2]
    z = rng.normal(size=(S, C)); z -= z.mean(axis=1, keepdims=True)
    theta = g("theta")[:, -1, :] + g("sigma_dev")[:, None] * math.sqrt(max(weeks, 0.0)) * z * math.sqrt(C / (C - 1))
    nu = rng.normal(size=(S, C)) * g("sigma_nu")[:, None] * math.sqrt(C / (C - 1)); nu -= nu.mean(axis=1, keepdims=True)
    mu = (theta + nu)[:, cars] + g("delta")[:, drv]                                   # draws x drivers
    m = g("m") if "m" in post.files else np.tile([-0.55, -0.45], (S, 1))
    steps = np.c_[np.zeros(S), m[:, 0], m[:, 0] + m[:, 1]]                            # Q1, Q2, Q3 track level
    sigma, pi, lc = g("sigma"), g("pi_bad"), g("loss_clean")
    lb = g("loss_bad") if "loss_bad" in post.files else lc + g("loss_gap")

    def best_laps(seg):
        shape = (S, D, laps_per_seg)
        bad = rng.uniform(size=shape) < pi[:, None, None]
        loss = rng.exponential(1.0, shape) * np.where(bad, lb[:, None, None], lc[:, None, None])
        laps = mu[:, :, None] + steps[:, seg, None, None] + rng.normal(size=shape) * sigma[:, None, None] + loss
        return laps.min(axis=2)

    out_n = (D - 10) // 2                                                             # 22 cars: 6 out in Q1 and Q2
    pos = np.zeros((S, D), dtype=int); alive = np.ones((S, D), dtype=bool)
    for seg, keep in [(0, D - out_n), (1, 10), (2, 0)]:
        t = np.where(alive, best_laps(seg), np.inf)
        order = np.argsort(t, axis=1)
        n_alive = alive.sum(axis=1)[0]
        ranks = np.empty_like(order); np.put_along_axis(ranks, order, np.arange(D)[None, :].repeat(S, 0), axis=1)
        if keep:
            out = alive & (ranks >= keep)
            pos[out] = (ranks[out] + 1)                                               # eliminated: keep their place
            alive &= ~out
        else:
            pos[alive] = ranks[alive] + 1
    return F, pos


def forecast(post, laps: pd.DataFrame, target: dict, weeks: float, **kw) -> dict:
    info = laps.groupby("driver").agg(name=("full_name", "first"), number=("driver_number", "first"))
    F, pos = simulate(post, weeks, **kw)
    D = pos.shape[1]
    rows = []
    for j, (d, c) in enumerate(F):
        code = str(post["drivers"][d]); p = pos[:, j]
        rows.append({"code": code, "name": str(info.loc[code, "name"]), "number": int(info.loc[code, "number"]),
                     "team": _short(post["cars"][c]), "p_pole": round(float((p == 1).mean()), 3),
                     "p_q3": round(float((p <= 10).mean()), 3), "p_q2": round(float((p <= D - (D - 10) // 2).mean()), 3),
                     "pos_median": int(np.median(p)), "pos_lo": int(np.percentile(p, 10)), "pos_hi": int(np.percentile(p, 90)),
                     "pos_dist": np.round(np.bincount(p, minlength=D + 1)[1:] / len(p), 3).tolist()})
    rows.sort(key=lambda r: (np.dot(r["pos_dist"], np.arange(1, D + 1)), -r["p_pole"]))
    return {"for_round": int(target["round"]), "name": target["name"], "location": target.get("location", ""),
            "date": str(target.get("date", ""))[:10], "made_after_round": int(post["events"][-1]),
            "weeks_ahead": round(weeks, 2), "sims": int(pos.shape[0]), "drivers": rows}


def next_target(post, laps: pd.DataFrame, schedule: Path | None) -> tuple[dict, float] | None:
    """The next round after the posterior's last one, from the schedule or (for history) the lap data."""
    last = int(post["events"][-1])
    t_last = pd.to_datetime(laps.loc[laps["round"] == last, "date_start"], format="ISO8601", utc=True).min()
    if schedule and Path(schedule).exists():
        for m in json.loads(Path(schedule).read_text()):
            if m["round"] == last + 1:
                when = pd.Timestamp(m["date"]).tz_convert("UTC") if pd.Timestamp(m["date"]).tzinfo else pd.Timestamp(m["date"], tz="UTC")
                return m, max(0.5, (when - t_last).total_seconds() / 604800)
    nxt = laps[laps["round"] == last + 1]
    if len(nxt):
        when = pd.to_datetime(nxt["date_start"], format="ISO8601", utc=True).min()
        r = nxt.iloc[0]
        return {"round": last + 1, "name": r["event"], "location": r.get("location", ""), "date": when.isoformat()}, (when - t_last).total_seconds() / 604800
    return None


def classification(laps: pd.DataFrame, rnd: int) -> list[str]:
    """Actual qualifying order: by furthest segment reached, then by best lap in that segment."""
    v = laps[(laps["round"] == rnd) & laps["lap_time"].notna() & ~laps["deleted"].astype(bool) & laps["segment"].notna()]
    if v.empty:
        return []
    seg_n = v["segment"].map({"Q1": 1, "Q2": 2, "Q3": 3})
    best = v.assign(seg_n=seg_n).groupby("driver").apply(
        lambda g: pd.Series({"seg": g["seg_n"].max(), "t": g.loc[g["seg_n"] == g["seg_n"].max(), "lap_time"].min()}),
        include_groups=False)
    return best.sort_values(["seg", "t"], ascending=[False, True]).index.tolist()


def score(fc: dict, actual: list[str]) -> dict | None:
    if not actual:
        return None
    rows = {r["code"]: r for r in fc["drivers"]}
    common = [c for c in actual if c in rows]
    if len(common) < 10:
        return None
    pole = actual[0]; q3 = set(actual[:10])
    p_pole = rows.get(pole, {}).get("p_pole", 0.0)
    brier = float(np.mean([(rows[c]["p_q3"] - (c in q3)) ** 2 for c in common]))
    base = float(np.mean([((10 / len(common)) - (c in q3)) ** 2 for c in common]))
    fpos = pd.Series({c: rows[c]["pos_median"] for c in common}).rank()
    apos = pd.Series({c: i for i, c in enumerate(common)}).rank()
    fav = max(rows.values(), key=lambda r: r["p_pole"])
    return {"round": fc["for_round"], "name": fc["name"], "actual_pole": pole, "p_actual_pole": p_pole,
            "favourite": fav["code"], "p_favourite": fav["p_pole"], "favourite_right": fav["code"] == pole,
            "q3_hits": int(sum(1 for c in common if rows[c]["p_q3"] >= 0.5 and c in q3)),
            "brier_q3": round(brier, 4), "brier_q3_baseline": round(base, 4),
            "log_score_pole": round(math.log(max(p_pole, 1e-3)), 3),
            "rank_corr": round(float(fpos.corr(apos)), 3)}


def scorecard(fc_dir: Path, laps: pd.DataFrame) -> dict:
    rounds = []
    for f in sorted(Path(fc_dir).glob("r*.json")):
        fc = json.loads(f.read_text())
        s = score(fc, classification(laps, fc["for_round"]))
        if s:
            rounds.append(s)
    files = sorted(Path(fc_dir).glob("r*.json"))
    out = {"rounds": rounds, "latest": files[-1].name if files else None}
    if rounds:
        b, b0 = np.mean([r["brier_q3"] for r in rounds]), np.mean([r["brier_q3_baseline"] for r in rounds])
        out["summary"] = {"forecasts": len(rounds), "favourite_right": int(sum(r["favourite_right"] for r in rounds)),
                          "q3_skill": round(1 - b / b0, 3), "mean_rank_corr": round(float(np.mean([r["rank_corr"] for r in rounds])), 3),
                          "mean_p_actual_pole": round(float(np.mean([r["p_actual_pole"] for r in rounds])), 3)}
    (Path(fc_dir) / "scorecard.json").write_text(json.dumps(out, indent=1))
    return out


def write(post, laps_csv: Path, out_dir: Path) -> Path | None:
    """Write forecasts/r<next>.json and refresh the scorecard. Returns the forecast path, if any."""
    laps = pd.read_csv(laps_csv)
    year = "".join(ch for ch in Path(laps_csv).stem if ch.isdigit())[-4:]
    tgt = next_target(post, laps, Path(laps_csv).parent / f"schedule_{year}.json")
    fc_dir = Path(out_dir) / "forecasts"; fc_dir.mkdir(parents=True, exist_ok=True)
    path = None
    if tgt:
        fc = forecast(post, laps, *tgt)
        path = fc_dir / f"r{fc['for_round']:02d}.json"
        path.write_text(json.dumps(fc, separators=(",", ":")))
        print(f"wrote {path} (round {fc['for_round']}: {fc['drivers'][0]['code']} favourite, {fc['drivers'][0]['p_pole']:.0%} pole)")
    scorecard(fc_dir, laps)
    return path
