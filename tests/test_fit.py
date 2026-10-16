import numpy as np
import pandas as pd

from purple_sectors.fit import gate, max_rhat


def test_gate_levels():
    assert gate(1, 8000, 1.002) == "pass"
    assert gate(45, 8000, 1.002) == "warn"         # 0.56% divergent
    assert gate(0, 8000, 1.016) == "warn"
    assert gate(0, 8000, 1.07) == "fail"
    assert gate(120, 8000, 1.0) == "fail"          # 1.5% divergent


class _Fit:
    chains, num_draws_sampling = 4, 500

    def __init__(self, shift_chain=0.0):
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame({"good": rng.normal(size=2000), "bad": rng.normal(size=2000)})
        self.df.loc[:499, "bad"] += shift_chain
        self.df.insert(0, "chain__", np.repeat(np.arange(1, 5), 500))

    def draws_pd(self, vars):
        return self.df


def test_max_rhat_names_the_worst_parameter():
    r, worst = max_rhat(_Fit(shift_chain=1.0), ["good", "bad"])
    assert worst == "bad" and r > 1.05
    r, _ = max_rhat(_Fit(), ["good", "bad"])
    assert r < 1.01
