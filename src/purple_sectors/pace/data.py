"""Qualifying push laps, several seasons → the design the pace model needs.

Vocabulary used everywhere in this package:
  season  a calendar year
  event   one qualifying session (season, round)
  car     a team in one season; the same team next year is a different car
  pair    a (driver, car) combination that actually happened
  base    one event × segment (Q1, Q2, Q3): the reference lap time it is compared to

Nothing here needs CmdStan or the network.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PUSH_THRESHOLD = 7.0          # % off the segment's best lap (the 107% rule)
SEGMENTS = {"Q1": 1, "Q2": 2, "Q3": 3}
WEEK_SECONDS = 7 * 24 * 3600
YOUNG_SEASONS = (1, 2)        # seasons after the debut that still get extra movement


def season_of(path) -> int | None:
    m = re.search(r"(20\d\d)", Path(path).stem)
    return int(m.group(1)) if m else None


def load_laps(paths, push_threshold: float = PUSH_THRESHOLD, keep_deleted: bool = True,
              dry_only: bool = True) -> pd.DataFrame:
    """Read data/raw/quali_laps_<year>.csv files and keep the push laps.

    A lap counts as a push lap when it is within `push_threshold` % of the quickest lap of
    its segment. Deleted laps (track limits) are kept by default: the car did go that fast.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "season" not in df:
            df["season"] = df["year"] if "year" in df else season_of(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date_start"] = pd.to_datetime(df["date_start"], format="ISO8601", utc=True)
    for col, default in (("deleted", False), ("is_pit_out_lap", False)):
        df[col] = df[col].fillna(default).astype(bool) if col in df else default
    ok = (df["lap_time"].notna() & (df["lap_time"] > 0) & df["segment"].isin(SEGMENTS)
          & df["driver"].notna() & df["team"].notna() & ~df["is_pit_out_lap"])
    if dry_only and "rainfall" in df:
        ok &= df["rainfall"].fillna(0) == 0
    if not keep_deleted:
        ok &= ~df["deleted"]
    v = df[ok].copy()
    best = v.groupby(["season", "round", "segment"])["lap_time"].transform("min")
    v["gap_pct"] = (v["lap_time"] / best - 1) * 100
    p = v[v["gap_pct"] <= push_threshold].copy()
    p["y"] = 100 * np.log(p["lap_time"])
    compound = p["compound"].astype(str).str.upper() if "compound" in p else pd.Series("SOFT", index=p.index)
    p["medium"] = (compound == "MEDIUM").astype(float)
    life = p["tyre_life"] if "tyre_life" in p else pd.Series(1.0, index=p.index)
    p["tyre_age"] = (life.fillna(1) - 1).clip(lower=0).astype(float)
    minutes = (p["date_start"] - p["date_start"].min()).dt.total_seconds() / 60
    p["t"] = (minutes - minutes.groupby([p["season"], p["round"], p["segment"]]).transform("mean")) / 10
    p["season"] = p["season"].astype(int)
    return p.sort_values(["season", "date_start"], kind="stable").reset_index(drop=True)


def load_debuts(path) -> dict:
    """{driver code: first F1 season}; written by the history fetch, optional."""
    path = Path(path) if path else None
    return json.loads(path.read_text()) if path and path.exists() else {}


@dataclass
class Design:
    laps: pd.DataFrame
    seasons: list            # [2023, 2024, ...]
    teams: dict              # season -> team names, in index order
    drivers: list            # driver codes, in index order
    events: pd.DataFrame     # one row per event, in model order (e = 1..E)
    pairs: list              # (driver, season, team) 0-based index triples, in pair order
    bases: list              # (season, round, segment) in base order
    debuts: dict
    season: np.ndarray       # the rest are per lap, 1-based like Stan
    rnd: np.ndarray
    team: np.ndarray
    driver: np.ndarray
    pair: np.ndarray
    base: np.ndarray
    event: np.ndarray
    t: np.ndarray
    medium: np.ndarray
    tyre_age: np.ndarray
    y: np.ndarray
    dt: np.ndarray           # S x RMAX, weeks since the previous round in that season
    y_ref: np.ndarray        # per base: median y
    present: np.ndarray      # D x S
    rookie: np.ndarray       # D x S: debut season
    young: np.ndarray        # D x S: second or third season

    @property
    def N(self): return len(self.y)
    @property
    def S(self): return len(self.seasons)
    @property
    def D(self): return len(self.drivers)
    @property
    def P(self): return len(self.pairs)
    @property
    def A(self): return len(self.bases)
    @property
    def E(self): return len(self.events)
    @property
    def T(self): return [len(self.teams[s]) for s in self.seasons]
    @property
    def R(self): return [int((self.events["season"] == s).sum()) for s in self.seasons]
    @property
    def TMAX(self): return max(self.T)
    @property
    def RMAX(self): return max(self.R)
    @property
    def dt_bar(self) -> float:
        positive = self.dt[self.dt > 0]
        return float(positive.mean()) if positive.size else 1.0

    def car_index(self, season: int, team: str) -> tuple[int, int]:
        """(season idx, team idx), 0-based."""
        return self.seasons.index(season), self.teams[season].index(team)

    def stan_data(self, y=None, prior_only: bool = False, save_loglik: bool = False) -> dict:
        return dict(
            N=self.N, S=self.S, TMAX=self.TMAX, RMAX=self.RMAX, T=self.T, R=self.R,
            D=self.D, P=self.P, A=self.A, E=self.E,
            season=self.season, rnd=self.rnd, team=self.team, driver=self.driver, pair=self.pair,
            base=self.base, event=self.event, t=self.t, medium=self.medium, tyre_age=self.tyre_age,
            y=self.y if y is None else np.asarray(y, dtype=float),
            dt=self.dt, dt_bar=self.dt_bar, y_ref=self.y_ref,
            present=self.present, rookie=self.rookie, young=self.young,
            prior_only=int(prior_only), save_loglik=int(save_loglik))


def build_design(laps: pd.DataFrame, debuts: dict | None = None) -> Design:
    p = laps
    seasons = sorted(int(s) for s in p["season"].unique())
    first = {"date": ("date_start", "min")}
    for col in ("event", "location"):
        if col in p:
            first[col] = (col, "first")
    ev = p.groupby(["season", "round"]).agg(**first).reset_index().sort_values(["season", "date"], kind="stable")
    ev["r"] = ev.groupby("season").cumcount() + 1
    ev["e"] = np.arange(1, len(ev) + 1)
    ev = ev.reset_index(drop=True)
    e_of = {(int(s), int(r)): int(e) for s, r, e in zip(ev["season"], ev["round"], ev["e"])}
    r_of = {(int(s), int(r)): int(k) for s, r, k in zip(ev["season"], ev["round"], ev["r"])}

    teams = {s: sorted(p.loc[p["season"] == s, "team"].unique()) for s in seasons}
    drivers = sorted(p["driver"].unique())
    d_of = {d: i for i, d in enumerate(drivers)}
    s_of = {s: i for i, s in enumerate(seasons)}

    season_i = np.array([s_of[int(s)] for s in p["season"]])
    team_i = np.array([teams[int(s)].index(tm) for s, tm in zip(p["season"], p["team"])])
    driver_i = np.array([d_of[d] for d in p["driver"]])
    pairs = sorted({(int(d), int(s), int(tm)) for d, s, tm in zip(driver_i, season_i, team_i)})
    pair_of = {pr: i for i, pr in enumerate(pairs)}
    pair_i = np.array([pair_of[(int(d), int(s), int(tm))] for d, s, tm in zip(driver_i, season_i, team_i)])

    bases = sorted({(int(s), int(r), g) for s, r, g in zip(p["season"], p["round"], p["segment"])},
                   key=lambda b: (e_of[(b[0], b[1])], SEGMENTS[b[2]]))
    base_of = {b: i for i, b in enumerate(bases)}
    base_i = np.array([base_of[(int(s), int(r), g)] for s, r, g in zip(p["season"], p["round"], p["segment"])])
    y = p["y"].to_numpy(dtype=float)
    y_ref = np.array([np.median(y[base_i == i]) for i in range(len(bases))])

    RMAX = int(ev.groupby("season").size().max())
    dt = np.zeros((len(seasons), RMAX))
    for s in seasons:
        dates = ev.loc[ev["season"] == s, "date"].to_numpy()
        gaps = np.diff(dates).astype("timedelta64[s]").astype(float) / WEEK_SECONDS
        dt[s_of[s], 1:1 + len(gaps)] = gaps

    debuts = dict(debuts or {})
    first_seen = p.groupby("driver")["season"].min().astype(int).to_dict()
    for d in drivers:  # no history file: a driver first seen after the window opens is a rookie then
        if d not in debuts and first_seen[d] > seasons[0]:
            debuts[d] = first_seen[d]
    present = np.zeros((len(drivers), len(seasons)), dtype=int)
    for d, s in p[["driver", "season"]].drop_duplicates().itertuples(index=False):
        present[d_of[d], s_of[int(s)]] = 1
    rookie = np.zeros_like(present)
    young = np.zeros_like(present)
    for d in drivers:
        if d in debuts:
            for s in seasons:
                rookie[d_of[d], s_of[s]] = int(s == debuts[d])
                young[d_of[d], s_of[s]] = int(s - debuts[d] in YOUNG_SEASONS)

    return Design(
        laps=p, seasons=seasons, teams=teams, drivers=drivers, events=ev, pairs=pairs, bases=bases,
        debuts=debuts,
        season=season_i + 1, rnd=np.array([r_of[(int(s), int(r))] for s, r in zip(p["season"], p["round"])]),
        team=team_i + 1, driver=driver_i + 1, pair=pair_i + 1, base=base_i + 1,
        event=np.array([e_of[(int(s), int(r))] for s, r in zip(p["season"], p["round"])]),
        t=p["t"].to_numpy(dtype=float), medium=p["medium"].to_numpy(dtype=float),
        tyre_age=p["tyre_age"].to_numpy(dtype=float), y=y, dt=dt, y_ref=y_ref,
        present=present, rookie=rookie, young=young)


def connectivity(des: Design) -> dict:
    """Are all drivers linked to each other through shared cars?

    Driver skill is only comparable across teams along chains of teammates and team moves.
    Drivers outside the main component can only be ranked against their own teammates.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for d, s, tm in des.pairs:
        union(("driver", d), ("car", s, tm))
    groups = {}
    for i, d in enumerate(des.drivers):
        groups.setdefault(find(("driver", i)), []).append(d)
    comps = sorted(groups.values(), key=len, reverse=True)
    return {"connected": len(comps) == 1, "components": comps,
            "cars_per_driver": {d: int(sum(1 for pr in des.pairs if pr[0] == i)) for i, d in enumerate(des.drivers)}}


def lap_files(laps_dir, years) -> list:
    files = [Path(laps_dir) / f"quali_laps_{y}.csv" for y in years]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit("missing lap files (run `python -m purple_sectors.fetch --year <year>`): " + ", ".join(missing))
    return files


def design_from_files(laps_dir, years, debuts_path=None, max_round: int | None = None, **load_kw) -> Design:
    laps = load_laps(lap_files(laps_dir, years), **load_kw)
    if max_round is not None:  # backfill: the latest season only up to this round
        last = laps["season"].max()
        laps = laps[(laps["season"] < last) | (laps["round"] <= max_round)].reset_index(drop=True)
    return build_design(laps, load_debuts(debuts_path))
