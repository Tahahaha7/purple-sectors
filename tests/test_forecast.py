import json

import numpy as np
import pandas as pd

import purple_sectors.forecast as FC
from helpers import laps_frame, posterior, write_laps


def test_simulated_grid_is_a_valid_qualifying_order(tmp_path):
    path, _ = posterior(tmp_path, extra=dict(m=np.tile([-0.5, -0.4], (400, 1))))
    F, pos = FC.simulate(np.load(path, allow_pickle=True), weeks=2.0, n=300)
    assert pos.shape == (300, len(F)) == (300, 6)
    assert all(sorted(row) == list(range(1, 7)) for row in pos)                    # every draw is a permutation


def test_forecast_probabilities_add_up(tmp_path):
    path, df = posterior(tmp_path, extra=dict(m=np.tile([-0.5, -0.4], (400, 1))))
    fc = FC.forecast(np.load(path, allow_pickle=True), df, {"round": 5, "name": "Next GP", "date": "2026-05-01"}, weeks=2.0)
    assert abs(sum(r["p_pole"] for r in fc["drivers"]) - 1) < 1e-6
    assert all(abs(sum(r["pos_dist"]) - 1) < 1e-2 and r["pos_lo"] <= r["pos_median"] <= r["pos_hi"] for r in fc["drivers"])
    assert fc["drivers"][0]["team"] == "Alpha"                                     # fastest car in the fake posterior


def test_classification_uses_furthest_segment_then_best_lap():
    df = laps_frame(1)
    df = df[~((df["driver"] == "FFF") & (df["segment"] != "Q1"))]                  # FFF knocked out in Q1
    df.loc[(df["driver"] == "EEE") & (df["segment"] == "Q3"), "lap_time"] = 80.0   # EEE takes pole in Q3
    order = FC.classification(df, 1)
    assert order[0] == "EEE" and order[-1] == "FFF"


def test_score_and_scorecard(tmp_path):
    path, df = posterior(tmp_path, extra=dict(m=np.tile([-0.5, -0.4], (400, 1))))
    post = np.load(path, allow_pickle=True)
    df5 = laps_frame(5); df5 = df5[df5["round"] == 5]
    laps = pd.concat([df, df5])
    fc = FC.forecast(post, laps, {"round": 5, "name": "Test 5 Grand Prix"}, weeks=2.0)
    (tmp_path / "forecasts").mkdir(); (tmp_path / "forecasts" / "r05.json").write_text(json.dumps(fc))
    card = FC.scorecard(tmp_path / "forecasts", laps)
    # a 6-car fake grid has fewer than 10 classified drivers, so the round is not scored, but the card is written
    assert card["latest"] == "r05.json" and (tmp_path / "forecasts" / "scorecard.json").exists()


def test_next_target_from_schedule_and_from_history(tmp_path):
    path, df = posterior(tmp_path)
    post = np.load(path, allow_pickle=True)
    sched = tmp_path / "schedule_2026.json"
    sched.write_text(json.dumps([{"round": 5, "name": "Next GP", "location": "X", "date": "2026-05-02T12:00:00+00:00"}]))
    tgt, weeks = FC.next_target(post, df, sched)
    assert tgt["round"] == 5 and 1.5 < weeks < 2.5
    df5 = pd.concat([df, laps_frame(5)[lambda d: d["round"] == 5]])
    tgt, weeks = FC.next_target(post, df5, None)                                     # history: next round is in the data
    assert tgt["round"] == 5 and abs(weeks - 2) < 0.1
