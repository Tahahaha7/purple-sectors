"""Fetch F1 qualifying laps (Q1/Q2/Q3) from the OpenF1 API into one CSV.

- Respects OpenF1's free tier (3 requests/s, 30 requests/min): every call is throttled,
  and rate-limit / server errors are retried with backoff (honouring Retry-After).
- Caches each finished session's raw responses in data/raw/openf1_cache/<session_key>/,
  so re-runs (and the scheduled GitHub Action) only download new sessions.
- Runs locally, in Colab (paste into a cell or `!python src/purple_sectors/fetch.py`),
  and in GitHub Actions. No API key needed.

Usage:  python src/purple_sectors/fetch.py --year 2026 [--strict]
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.openf1.org/v1"
MIN_INTERVAL = 60 / 27            # seconds between calls: stays under 30/min
MAX_RETRIES = 8
FRESH_HOURS = 6                   # don't cache sessions that ended this recently
SESSION_ENDPOINTS = ["laps", "race_control", "drivers", "stints", "weather"]


# ---- API client ------------------------------------------------------------
class OpenF1:
    def __init__(self, cache_dir: Path):
        self.cache = Path(cache_dir)
        self.http = requests.Session()
        self.last = 0.0
        self.calls = 0

    def _throttle(self):
        wait = self.last + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()

    def get(self, endpoint: str, **params) -> list:
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle()
            self.calls += 1
            try:
                r = self.http.get(f"{BASE}/{endpoint}", params=params, timeout=60)
            except requests.RequestException as e:
                wait = min(60, 5 * 2 ** attempt)
                print(f"     network error on /{endpoint} ({e.__class__.__name__}), retry in {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 404:                      # OpenF1: "no results"
                return []
            if r.status_code in (429, 500, 502, 503, 504):
                try:
                    wait = float(r.headers.get("Retry-After", ""))
                except ValueError:
                    wait = min(90, 10 * 2 ** (attempt - 1))
                wait += random.uniform(0, 2)
                print(f"     {r.status_code} on /{endpoint}, waiting {wait:.0f}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"gave up on /{endpoint} after {MAX_RETRIES} attempts")

    def session(self, sk: int, fresh: bool) -> dict:
        """Raw responses for one session, from cache when available."""
        folder = self.cache / str(sk)
        if (folder / "_complete").exists():
            return {ep: json.loads((folder / f"{ep}.json").read_text()) for ep in SESSION_ENDPOINTS}
        raw = {ep: self.get(ep, session_key=sk) for ep in SESSION_ENDPOINTS}
        if raw["laps"] and not fresh:                     # never freeze a still-filling feed
            folder.mkdir(parents=True, exist_ok=True)
            for ep, data in raw.items():
                (folder / f"{ep}.json").write_text(json.dumps(data))
            (folder / "_complete").touch()
        return raw


# ---- per-session processing --------------------------------------------------
def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="ISO8601")


def to_sec(t: str) -> float:
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def segment_bounds(rc: pd.DataFrame):
    """End of Q1 and end of Q2, from chequered flags (fallback: qualifying_phase)."""
    flags = sorted(rc.loc[rc["message"].str.contains("CHEQUERED FLAG", na=False), "date"])
    if len(flags) >= 3:
        return flags[0], flags[1]
    if "qualifying_phase" in rc and rc["qualifying_phase"].notna().any():
        starts = rc.dropna(subset=["qualifying_phase"]).groupby("qualifying_phase")["date"].min()
        if 2 in starts and 3 in starts:
            return starts[2], starts[3]
    return None


def mark_deleted(laps: pd.DataFrame, rc: pd.DataFrame) -> pd.Series:
    """Flag deleted laps by matching race-control messages on lap time or lap number."""
    deleted = pd.Series(False, index=laps.index)
    for msg in rc.sort_values("date")["message"].dropna():
        if "DELETED" not in msg and "REINSTATED" not in msg:
            continue
        car = re.search(r"CAR (\d+)", msg)
        if not car:
            continue
        drv = laps["driver_number"] == int(car.group(1))
        hit = pd.Series(False, index=laps.index)
        if t := re.search(r"TIME (\d+:\d+\.\d+)", msg):
            hit |= drv & ((laps["lap_duration"] - to_sec(t.group(1))).abs() < 0.001)
        if n := re.search(r"\bLAP (\d+)\s+\d", msg):
            hit |= drv & (laps["lap_number"] == int(n.group(1)))
        deleted[hit] = "DELETED" in msg
    return deleted


def add_tyres(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    laps["compound"], laps["tyre_life"] = None, float("nan")
    for _, s in stints.iterrows():
        end = s["lap_end"] if pd.notna(s["lap_end"]) else 999
        m = (laps["driver_number"] == s["driver_number"]) & laps["lap_number"].between(s["lap_start"], end)
        laps.loc[m, "compound"] = s["compound"]
        laps.loc[m, "tyre_life"] = s["tyre_age_at_start"] + laps.loc[m, "lap_number"] - s["lap_start"]
    return laps


def add_weather(laps: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    if w.empty:
        return laps
    w = w[["date", "air_temperature", "track_temperature", "rainfall"]].copy()
    w["date"] = to_dt(w["date"])
    has = laps["date_start"].notna()
    merged = pd.merge_asof(laps[has].sort_values("date_start"), w.sort_values("date"),
                           left_on="date_start", right_on="date", direction="backward")
    return pd.concat([merged.drop(columns="date"), laps[~has]], ignore_index=True)


def build_session(raw: dict) -> pd.DataFrame:
    laps = pd.DataFrame(raw["laps"])
    if laps.empty:
        raise ValueError("no laps")
    laps["date_start"] = to_dt(laps["date_start"])

    rc = pd.DataFrame(raw["race_control"])
    bounds = None
    if not rc.empty:
        rc["date"] = to_dt(rc["date"])
        laps["deleted"] = mark_deleted(laps, rc)
        bounds = segment_bounds(rc)
    else:
        laps["deleted"] = False

    laps["segment"] = None
    if bounds:
        b1, b2 = bounds
        laps["segment"] = "Q3"
        laps.loc[laps["date_start"] < b2, "segment"] = "Q2"
        laps.loc[laps["date_start"] < b1, "segment"] = "Q1"
        laps.loc[laps["date_start"].isna(), "segment"] = None
    else:
        print("     warning: could not split Q1/Q2/Q3")

    drivers = pd.DataFrame(raw["drivers"]).drop_duplicates("driver_number")
    laps = laps.merge(drivers[["driver_number", "name_acronym", "full_name", "team_name"]],
                      on="driver_number", how="left")
    laps = add_tyres(laps, pd.DataFrame(raw["stints"]))
    return add_weather(laps, pd.DataFrame(raw["weather"]))


# ---- season ------------------------------------------------------------------
COLUMNS = ["round", "event", "location", "country", "circuit", "session_key", "segment",
           "driver", "driver_number", "full_name", "team", "lap_number", "lap_time",
           "s1", "s2", "s3", "date_start", "is_pit_out_lap", "compound", "tyre_life",
           "deleted", "i1_speed", "i2_speed", "st_speed", "air_temp", "track_temp", "rainfall"]


def fetch_season(year: int, out: Path, cache_dir: Path) -> tuple[pd.DataFrame, list]:
    api = OpenF1(cache_dir)
    meetings = pd.DataFrame(api.get("meetings", year=year))
    if meetings.empty:
        raise SystemExit("OpenF1 returned no meetings. Check access to api.openf1.org.")
    meetings = meetings[~meetings["meeting_name"].str.contains("Testing", case=False)]
    meetings = meetings.sort_values("date_start").reset_index(drop=True)
    meetings["round"] = meetings.index + 1

    sessions = pd.DataFrame(api.get("sessions", year=year, session_name="Qualifying"))
    sessions["date_end"] = to_dt(sessions["date_end"])
    now = pd.Timestamp.now(tz="UTC")
    sessions = sessions[sessions["date_end"] < now]
    sessions = sessions.merge(meetings[["meeting_key", "meeting_name", "round"]], on="meeting_key")
    sessions = sessions.sort_values("round")
    todo = sum(not (Path(cache_dir) / str(k) / "_complete").exists() for k in sessions["session_key"])
    print(f"{len(sessions)} qualifying sessions · {todo} to download (~{todo * 5 * MIN_INTERVAL / 60:.0f} min)")

    frames, skipped = [], []
    for _, s in sessions.iterrows():
        tag = f"R{s['round']:<2} {s['meeting_name']}"
        try:
            fresh = now - s["date_end"] < pd.Timedelta(hours=FRESH_HOURS)
            df = build_session(api.session(int(s["session_key"]), fresh))
            for col, val in [("round", s["round"]), ("event", s["meeting_name"]),
                             ("location", s.get("location")), ("country", s.get("country_name")),
                             ("circuit", s.get("circuit_short_name"))]:
                df[col] = val
            frames.append(df)
            print(f"ok   {tag}  ({len(df)} laps)")
        except Exception as e:
            skipped.append(tag)
            print(f"skip {tag}: {type(e).__name__}: {e}")

    if not frames:
        raise SystemExit("No laps loaded.")
    data = pd.concat(frames, ignore_index=True).rename(columns={
        "name_acronym": "driver", "team_name": "team", "lap_duration": "lap_time",
        "duration_sector_1": "s1", "duration_sector_2": "s2", "duration_sector_3": "s3",
        "air_temperature": "air_temp", "track_temperature": "track_temp"})
    data = data[[c for c in COLUMNS if c in data]].sort_values(["round", "date_start"])
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False)
    print(f"\nSaved {out}: {len(data)} laps, {data['round'].nunique()} events, "
          f"{api.calls} API calls" + (f", SKIPPED: {', '.join(skipped)}" if skipped else ""))
    return data, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/openf1_cache"))
    ap.add_argument("--strict", action="store_true", help="exit 1 if any session was skipped (CI)")
    args, _ = ap.parse_known_args()                       # tolerates Jupyter's own arguments
    out = args.out or Path(f"data/raw/quali_laps_{args.year}.csv")
    _, skipped = fetch_season(args.year, out, args.cache_dir)
    if args.strict and skipped:
        sys.exit(1)
