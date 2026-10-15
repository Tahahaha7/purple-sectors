"""Turn raw OpenF1 qualifying laps into model-ready push laps."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

PUSH_THRESHOLD = 7.0  # % off segment best; push and cool laps separate cleanly here
MIN_EVENTS = 3        # drop part-time / substitute drivers
SEG_INDEX = {"Q1": 1, "Q2": 2, "Q3": 3}


def one_car_per_driver(p: pd.DataFrame, min_events: int = MIN_EVENTS) -> pd.DataFrame:
    """Keep each driver only in their main team, and only if they ran >= min_events events.
    Every remaining driver then has exactly one car all season."""
    n = p.groupby(["driver", "team"])["round"].nunique().reset_index(name="events")
    main = n.sort_values("events").groupby("driver").tail(1)
    main = main[main["events"] >= min_events]
    dropped = n.merge(main, how="left", indicator=True).query("_merge == 'left_only'")
    for _, r in dropped.iterrows():
        print(f"  excluded {r.driver} @ {r.team} ({r.events} events)")
    return p.merge(main[["driver", "team"]], on=["driver", "team"])


def load_push_laps(path: str, single_car: bool = True, max_round: int | None = None) -> pd.DataFrame:
    """Push laps, optionally only up to (and including) max_round, for season-history snapshots."""
    df = pd.read_csv(path)
    if max_round is not None:
        df = df[df["round"] <= max_round]
    df["date_start"] = pd.to_datetime(df["date_start"], format="ISO8601", utc=True)
    valid = (df["lap_time"].notna() & ~df["deleted"] & ~df["is_pit_out_lap"]
             & df["segment"].notna() & (df["rainfall"].fillna(0) == 0))
    v = df[valid].copy()
    seg_best = v.groupby(["round", "segment"])["lap_time"].transform("min")
    v["gap_pct"] = (v["lap_time"] / seg_best - 1) * 100
    p = v[v["gap_pct"] <= PUSH_THRESHOLD].copy()
    p["medium"] = (p["compound"] == "MEDIUM").astype(float)
    p["tyre_age"] = (p["tyre_life"].fillna(1) - 1).clip(lower=0)
    p["y"] = 100 * np.log(p["lap_time"])
    # minutes since segment's mean lap start, in units of 10 min (v1.1 track evolution)
    t = (p["date_start"] - p["date_start"].min()).dt.total_seconds() / 60
    p["t_seg"] = (t - t.groupby([p["round"], p["segment"]]).transform("mean")) / 10
    if single_car:   # early in the season "3 events" is too strict: require ceil(rounds / 3), at most 3
        p = one_car_per_driver(p, min(MIN_EVENTS, -(-p["round"].nunique() // 3)))
    return p.reset_index(drop=True)


@dataclass
class Design:
    """Index structure of the data: who drove what, where, in which segment (0-based)."""
    laps: pd.DataFrame
    drivers: list
    cars: list
    events: list
    driver: np.ndarray
    car: np.ndarray
    event: np.ndarray
    seg: np.ndarray
    medium: np.ndarray
    tyre_age: np.ndarray
    dt: np.ndarray
    t_seg: np.ndarray

    @property
    def N(self): return len(self.driver)
    @property
    def D(self): return len(self.drivers)
    @property
    def C(self): return len(self.cars)
    @property
    def E(self): return len(self.events)

    def stan_data(self, y: np.ndarray, prior_only: bool = False) -> dict:
        y_ref = pd.Series(y).groupby(self.event).median().to_numpy()
        return dict(N=self.N, D=self.D, C=self.C, E=self.E,
                    driver=self.driver + 1, car=self.car + 1, event=self.event + 1,
                    seg=self.seg + 1, medium=self.medium, tyre_age=self.tyre_age,
                    dt=self.dt, t_seg=self.t_seg, y_ref=y_ref, y=y,
                    prior_only=int(prior_only))


def build_design(p: pd.DataFrame) -> Design:
    starts = p.groupby("round")["date_start"].min().sort_values()
    events = list(starts.index)
    secs = starts.diff().dt.total_seconds().fillna(0).to_numpy()
    drivers = sorted(p["driver"].unique())
    cars = sorted(p["team"].unique())
    return Design(
        laps=p, drivers=drivers, cars=cars, events=events,
        driver=p["driver"].map({d: i for i, d in enumerate(drivers)}).to_numpy(),
        car=p["team"].map({c: i for i, c in enumerate(cars)}).to_numpy(),
        event=p["round"].map({r: i for i, r in enumerate(events)}).to_numpy(),
        seg=p["segment"].map(SEG_INDEX).to_numpy() - 1,
        medium=p["medium"].to_numpy(), tyre_age=p["tyre_age"].to_numpy(),
        dt=secs / 604800, t_seg=p["t_seg"].to_numpy())
