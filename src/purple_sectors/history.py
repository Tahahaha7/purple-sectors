"""Earlier seasons as background evidence: qualifying results 2022-2025 from Jolpica (the Ergast successor),
plus the teammate network that links every driver.

Usage:  python -m purple_sectors.history --from 2022 --to 2025 --laps data/raw/quali_laps_2026.csv

Writes data/raw/history_quali_2022_2025.csv (one row per driver per round: best Q1/Q2/Q3 time)
and site/data/network.json (drivers, and an edge for every season two drivers shared a team).
"""
import argparse
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.jolpi.ca/ergast/f1"
PAGE = 100
MIN_INTERVAL = 0.6                      # Jolpica: a few requests per second, 500 per hour


def _get(session, url, params, tries=6):
    for k in range(tries):
        r = session.get(url, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 5 * 2 ** k)); continue
        r.raise_for_status()
        time.sleep(MIN_INTERVAL)
        return r.json()
    raise RuntimeError(f"gave up on {url} {params}")


def to_sec(t) -> float | None:
    if not t or not isinstance(t, str) or ":" not in t:
        return None
    m, s = t.split(":")
    try:
        return int(m) * 60 + float(s)
    except ValueError:
        return None


def fetch_season(season: int, session=None) -> pd.DataFrame:
    """Every driver's qualifying result for every round of a season (paged)."""
    session = session or requests.Session()
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        js = _get(session, f"{BASE}/{season}/qualifying.json", {"limit": PAGE, "offset": offset})
        mr = js["MRData"]; total = int(mr["total"])
        for race in mr["RaceTable"]["Races"]:
            loc = race.get("Circuit", {}).get("Location", {})
            for q in race.get("QualifyingResults", []):
                d, c = q["Driver"], q["Constructor"]
                rows.append({"season": int(race["season"]), "round": int(race["round"]), "race": race.get("raceName"),
                             "circuit": race.get("Circuit", {}).get("circuitId"), "location": loc.get("locality"),
                             "date": race.get("date"), "driver": d.get("code") or d["driverId"][:3].upper(),
                             "driver_id": d["driverId"], "name": f"{d.get('givenName', '')} {d.get('familyName', '')}".strip(),
                             "number": d.get("permanentNumber"), "team_id": c["constructorId"], "team": c["name"],
                             "position": int(q["position"]), "q1": to_sec(q.get("Q1")), "q2": to_sec(q.get("Q2")),
                             "q3": to_sec(q.get("Q3"))})
        offset += PAGE
    return pd.DataFrame(rows)


def teammate_edges(hist: pd.DataFrame, laps2026: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (season, team, driver pair) with the number of rounds they both qualified."""
    frames = [hist[["season", "round", "team", "driver"]]]
    if laps2026 is not None and len(laps2026):
        t = laps2026.drop_duplicates(["round", "team", "driver"])[["round", "team", "driver"]].assign(season=2026)
        frames.append(t[["season", "round", "team", "driver"]])
    df = pd.concat(frames)
    pairs = defaultdict(int)
    for (season, rnd, team), g in df.groupby(["season", "round", "team"]):
        ds = sorted(set(g["driver"]))
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                pairs[(season, team, ds[i], ds[j])] += 1
    return pd.DataFrame([{"season": s, "team": t, "a": a, "b": b, "rounds": n} for (s, t, a, b), n in pairs.items()])


def components(edges: pd.DataFrame, drivers: list[str]) -> list[set]:
    adj = defaultdict(set)
    for a, b in zip(edges["a"], edges["b"]):
        adj[a].add(b); adj[b].add(a)
    seen, comps = set(), []
    for d in drivers:
        if d in seen:
            continue
        comp, q = set(), deque([d])
        while q:
            x = q.popleft()
            if x in comp:
                continue
            comp.add(x); q.extend(adj[x] - comp)
        seen |= comp; comps.append(comp & set(drivers))
    return comps


def network(hist: pd.DataFrame, laps2026: pd.DataFrame) -> dict:
    edges = teammate_edges(hist, laps2026)
    grid = sorted(laps2026["driver"].unique()) if len(laps2026) else []
    names = {**hist.drop_duplicates("driver").set_index("driver")["name"].to_dict(),
             **(laps2026.drop_duplicates("driver").set_index("driver")["full_name"].to_dict() if len(laps2026) else {})}
    teams = defaultdict(list)
    for (d, s), g in pd.concat([hist[["driver", "season", "team"]],
                                laps2026.assign(season=2026)[["driver", "season", "team"]] if len(laps2026) else hist.iloc[:0][["driver", "season", "team"]]]
                               ).drop_duplicates().groupby(["driver", "season"]):
        teams[d].append({"season": int(s), "teams": sorted(g["team"].unique())})
    comps = components(edges, grid)
    return {"grid_2026": grid,
            "nodes": [{"id": d, "name": names.get(d, d), "on_2026_grid": d in grid, "seasons": teams[d]} for d in sorted(teams)],
            "edges": edges.sort_values(["season", "team", "a"]).to_dict("records"),
            "connected": len(comps) == 1, "components": [sorted(c) for c in comps]}


def main(first: int, last: int, laps: Path, out_csv: Path, out_net: Path) -> dict:
    s = requests.Session()
    hist = pd.concat([fetch_season(y, s) for y in range(first, last + 1)], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True); hist.to_csv(out_csv, index=False)
    l26 = pd.read_csv(laps) if Path(laps).exists() else pd.DataFrame()
    net = network(hist, l26)
    out_net.parent.mkdir(parents=True, exist_ok=True); out_net.write_text(json.dumps(net, indent=1))
    print(f"{len(hist)} results, {hist['driver'].nunique()} drivers, {len(net['edges'])} teammate pairings; "
          f"2026 grid connected: {net['connected']} ({len(net['components'])} component(s))")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        lines = [f"## History {first}–{last}", f"{len(hist)} qualifying results, {hist['driver'].nunique()} drivers, {len(net['edges'])} teammate pairings.",
                 f"**2026 grid connected through teammates: {'✅ yes' if net['connected'] else '❌ no'}**"]
        lines += [f"- component: {', '.join(c)}" for c in net["components"]]
        Path(summary).open("a").write("\n".join(lines) + "\n")
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="first", type=int, default=2022)
    ap.add_argument("--to", dest="last", type=int, default=2025)
    ap.add_argument("--laps", type=Path, default=Path("data/raw/quali_laps_2026.csv"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--network", type=Path, default=Path("site/data/network.json"))
    a, _ = ap.parse_known_args()
    out = a.out or Path(f"data/raw/history_quali_{a.first}_{a.last}.csv")
    main(a.first, a.last, a.laps, out, a.network)
