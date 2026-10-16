import json
from pathlib import Path

import pandas as pd

import purple_sectors.fetch as F

UTC = lambda s: pd.Timestamp(s, tz="UTC")


def test_mark_deleted_by_time_lap_number_and_reinstated():
    laps = pd.DataFrame({"driver_number": [23, 23, 43, 43, 14], "lap_number": [3, 4, 6, 7, 9],
                         "lap_duration": [102.439, 101.9, 130.089, 136.95, None]})
    rc = pd.DataFrame({"date": [UTC("2026-08-22 16:05"), UTC("2026-08-22 16:14"), UTC("2026-08-22 16:15"),
                                UTC("2026-08-22 16:16"), UTC("2026-08-22 16:18")],
                       "message": ["CAR 23 (ALB) TIME 1:42.439 DELETED - TRACK LIMITS AT TURN 11 LAP 3 16:03:41",
                                   "CAR 43 (COL) TIME 2:10.089 DELETED - TRACK LIMITS AT TURN 15 LAP 6 16:13:16",
                                   "CAR 43 (COL) TIME 2:16.950 DELETED - TRACK LIMITS AT TURN 15 LAP 7 16:15:31",
                                   "CAR 43 (COL) TIME 2:16.950 REINSTATED",
                                   "CAR 14 (ALO) LAP  DELETED - TRACK LIMITS AT TURN 15 LAP 9 16:18:13 (PIT)"]})
    assert F.mark_deleted(laps, rc).tolist() == [True, False, True, False, True]


def test_segment_bounds_from_chequered_flags():
    rc = pd.DataFrame({"date": [UTC("2026-08-22 16:18"), UTC("2026-08-22 16:45"), UTC("2026-08-22 17:05")],
                       "message": ["CHEQUERED FLAG"] * 3})
    assert F.segment_bounds(rc) == (UTC("2026-08-22 16:18"), UTC("2026-08-22 16:45"))


def _payload(end, empty=False):
    t0 = end - pd.Timedelta(minutes=60)
    iso = lambda m: (t0 + pd.Timedelta(minutes=m)).isoformat()
    if empty:
        return dict(laps=[], race_control=[], drivers=[], stints=[], weather=[])
    return dict(
        laps=[dict(driver_number=1, lap_number=k + 2, lap_duration=80 - k * 0.3, date_start=iso(m), is_pit_out_lap=False,
                   duration_sector_1=25, duration_sector_2=27, duration_sector_3=27, i1_speed=1, i2_speed=1, st_speed=1)
              for k, m in enumerate([5, 20, 40])],
        race_control=[dict(date=iso(m), message="CHEQUERED FLAG") for m in (18, 35, 55)],
        drivers=[dict(driver_number=1, name_acronym="VER", full_name="Max VERSTAPPEN", team_name="Red Bull Racing")],
        stints=[dict(driver_number=1, stint_number=1, lap_start=1, lap_end=None, compound="SOFT", tyre_age_at_start=0)],
        weather=[dict(date=iso(0), air_temperature=20, track_temperature=30, rainfall=0)])


class _Resp:
    def __init__(self, code, data, headers=None):
        self.status_code, self._d, self.headers = code, data, headers or {}

    def json(self):
        return self._d

    def raise_for_status(self):
        pass


def test_fetch_season_cancelled_rounds_numbering_and_cache(tmp_path):
    now = pd.Timestamp.now(tz="UTC"); ago = lambda d: now - pd.Timedelta(days=d)
    M = [(10, "Pre-Season Testing", 200, None), (11, "Australian Grand Prix", 180, (1, False)),
         (12, "Bahrain Grand Prix (April)", 150, (2, True)), (13, "Miami Grand Prix", 120, (3, False)),
         (14, "Singapore Grand Prix", 3, (4, False)), (15, "Late-data Grand Prix", 1.5, (5, True)),
         (16, "Future Grand Prix", -10, (6, False))]
    payloads = {sk: _payload(ago(d), empty) for _, _, d, v in M if v for sk, empty in [v]}
    calls, first = [], {"429": True}

    def fake_get(self, url, params=None, timeout=None):
        ep = url.rsplit("/", 1)[1]; calls.append(ep)
        if ep == "meetings":
            return _Resp(200, [dict(meeting_key=k, meeting_name=n, date_start=(ago(d) - pd.Timedelta(days=2)).isoformat()) for k, n, d, _ in M])
        if ep == "sessions":
            return _Resp(200, [dict(session_key=v[0], meeting_key=k, date_end=ago(d).isoformat(), location="X", country_name="Y",
                                    circuit_short_name="Z") for k, n, d, v in M if v])
        if ep == "laps" and first["429"]:
            first["429"] = False; return _Resp(429, None, {"Retry-After": "0"})          # one rate-limit hit
        return _Resp(200, payloads[params["session_key"]][ep])

    orig_get, orig_sleep, orig_int = F.requests.Session.get, F.time.sleep, F.MIN_INTERVAL
    F.requests.Session.get, F.time.sleep, F.MIN_INTERVAL = fake_get, (lambda s: None), 0.0
    try:
        data, skipped = F.fetch_season(2026, tmp_path / "out.csv", tmp_path / "cache")
        assert dict(data.groupby("round")["event"].first()) == {1: "Australian Grand Prix", 2: "Miami Grand Prix", 3: "Singapore Grand Prix"}
        assert skipped == ["R4  Late-data Grand Prix"]                                # recent and empty: an error, not a cancellation
        assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["1", "2", "3", "4"]   # session 5 not frozen
        calls.clear()
        F.fetch_season(2026, tmp_path / "out.csv", tmp_path / "cache")
        assert calls.count("laps") == 1                                               # only the unfinished session is refetched
    finally:
        F.requests.Session.get, F.time.sleep, F.MIN_INTERVAL = orig_get, orig_sleep, orig_int
