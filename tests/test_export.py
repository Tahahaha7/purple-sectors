import json

import numpy as np

import purple_sectors.export as X
from helpers import posterior, write_laps


def test_density_integrates_to_one_and_matches_probability():
    rng = np.random.default_rng(0)
    for mu, sd in [(-0.39, 0.06), (-0.05, 0.05), (0.02, 0.2)]:
        d = rng.normal(mu, sd, 20000)
        x = np.linspace(*X.GRID[:2], X.GRID[2]); f = np.array(X.density(d))
        assert abs(np.trapezoid(f, x) - (np.abs(d) < 0.8).mean()) < 0.02
        m = x <= 0
        assert abs(np.trapezoid(f[m], x[m]) - (d < 0).mean()) < 0.02


def test_outputs_are_compact_float64_rounded(tmp_path):
    path, _ = posterior(tmp_path)
    post = np.load(path, allow_pickle=True)
    q = X._q(post["theta"][:, -1, 0].astype("float32"))
    assert all(len(repr(v)) <= 7 for v in q)                                       # no float32 noise like -1.2339999675


def test_snapshot_index_and_draws(tmp_path):
    path, df = posterior(tmp_path)
    laps = write_laps(tmp_path, df)
    X.main(path, laps, tmp_path / "site", snapshot_dir=tmp_path / "snaps")
    snap = json.loads((tmp_path / "snaps" / "r04.json").read_text())
    assert {"cars", "drivers", "teammates", "events", "params", "car_share", "grid"} <= set(snap)
    assert all(len(t["density"]) == X.GRID[2] for t in snap["teammates"])
    assert all(t["gap"] <= 0 and 0.5 <= t["p"] <= 1 for t in snap["teammates"])    # oriented: first driver quicker
    idx = json.loads((tmp_path / "snaps" / "index.json").read_text())
    assert [r["file"] for r in idx["rounds"]] == ["r04.json"]                        # *_draws.json are not rounds
    draws = json.loads((tmp_path / "snaps" / "r04_draws.json").read_text())
    assert draws["n"] == 150 and all(len(v) == 150 for v in draws["cars"].values())
    soft = json.loads((tmp_path / "site" / "soft.json").read_text())
    assert "density" not in soft["pairs"][0] and len(soft["pairs"][0]["dots"]) == 20


def test_status_fallbacks(tmp_path):
    path, _ = posterior(tmp_path, extra=dict(status=np.array("warn"), passed=np.array(False)))
    assert X._status(np.load(path)) == "warn"
    path, _ = posterior(tmp_path, extra=dict(passed=np.array(False)))
    assert X._status(np.load(path)) == "fail"
    path, _ = posterior(tmp_path)
    assert X._status(np.load(path)) == "pass"
