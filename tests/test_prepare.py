import numpy as np
import pandas as pd

from helpers import laps_frame, write_laps
from purple_sectors.prepare import PUSH_THRESHOLD, build_design, load_push_laps, one_car_per_driver


def test_push_lap_filter_drops_invalid_and_slow_laps(tmp_path):
    df = laps_frame(2)
    df.loc[0, "deleted"] = True
    df.loc[1, "is_pit_out_lap"] = True
    df.loc[2, "rainfall"] = 1
    df.loc[3, "lap_time"] = df.loc[3, "lap_time"] * (1 + (PUSH_THRESHOLD + 3) / 100)   # a cool-down lap
    p = load_push_laps(str(write_laps(tmp_path, df)))
    assert len(p) == len(df) - 4
    assert (p["gap_pct"] <= PUSH_THRESHOLD).all()
    assert np.allclose(p["y"], 100 * np.log(p["lap_time"]))


def test_one_car_per_driver_keeps_main_team_and_drops_substitutes():
    df = laps_frame(4)
    sub = df[(df["driver"] == "AAA") & (df["round"] == 4)].copy()
    sub["team"] = "Bravo"                              # AAA subs for Bravo once
    df = pd.concat([df[~((df["driver"] == "AAA") & (df["round"] == 4))], sub])
    one = df[(df["driver"] == "FFF") & (df["round"] == 1)].assign(driver="ZZZ")   # a one-off driver
    out = one_car_per_driver(pd.concat([df, one]), min_events=3)
    assert set(out.loc[out["driver"] == "AAA", "team"]) == {"Alpha"}
    assert "ZZZ" not in set(out["driver"])
    assert out.groupby("driver")["team"].nunique().max() == 1


def test_early_season_rule_keeps_a_driver_who_missed_round_1(tmp_path):
    df = laps_frame(6)
    df = df[~((df["driver"] == "AAA") & (df["round"] == 1))]                     # no valid laps in round 1
    path = write_laps(tmp_path, df)
    for k in (2, 3):                                                               # needs ceil(k/3) = 1 event
        assert "AAA" in set(load_push_laps(str(path), max_round=k)["driver"])
    full = load_push_laps(str(path))                                               # 6 rounds: needs 2 events, has 5
    assert "AAA" in set(full["driver"])


def test_design_indices_and_time_covariate(tmp_path):
    des = build_design(load_push_laps(str(write_laps(tmp_path, laps_frame(3)))))
    assert (des.N, des.D, des.C, des.E) == (len(des.laps), 6, 3, 3)
    assert des.dt[0] == 0 and np.allclose(des.dt[1:], 2.0, atol=0.02)          # two weeks apart
    seg_mean = pd.Series(des.t_seg).groupby([des.event, des.seg]).mean()
    assert np.allclose(seg_mean, 0, atol=1e-9)                                   # centred per segment
    data = des.stan_data(des.laps["y"].to_numpy())
    assert data["driver"].min() == 1 and data["driver"].max() == data["D"]
    assert len(data["y_ref"]) == data["E"] and len(data["t_seg"]) == data["N"]
