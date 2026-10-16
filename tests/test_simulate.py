import numpy as np

from helpers import laps_frame, write_laps
from purple_sectors.prepare import build_design, load_push_laps
from purple_sectors.simulate import draw_params, simulate_laps, summaries


def test_generative_model_shapes_and_constraints(tmp_path):
    des = build_design(load_push_laps(str(write_laps(tmp_path, laps_frame(4)))))
    rng = np.random.default_rng(3)
    p = draw_params(des, rng, v12=True)
    assert p["theta"].shape == (des.E, des.C) and np.allclose(p["theta"].sum(axis=1), 0, atol=1e-9)
    assert np.allclose(p["nu"].sum(axis=1), 0, atol=1e-9) and abs(p["delta"].sum()) < 1e-9
    assert 0 < p["pi_bad"] < 1 and p["loss_clean"] >= 0.01
    y = simulate_laps(des, p, rng)
    assert y.shape == (des.N,) and np.isfinite(y).all()
    s = summaries(des, y)
    assert set(s) == {"field_spread", "teammate_gap", "q1_q2_step", "lap_skew", "lap_kurt"}
