"""Parameter recovery: simulate a season from known parameters on the real design, fit, check coverage.

Usage:  python -m purple_sectors.recovery --stan stan/quali_v13.stan --report recovery.json
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel

from purple_sectors.prepare import build_design, load_push_laps
from purple_sectors.simulate import draw_params, simulate_laps

TRUTH = dict(m=np.array([-0.55, -0.40]), sigma_kappa=0.15, mu_gamma=-0.4, sigma_gamma=0.15,
             sigma_car=1.2, sigma_driver=0.15, beta_m=0.3, beta_a=0.03, sigma=0.2,
             pi_bad=0.12, loss_clean=0.28, loss_gap=1.2)


def flags_for(stan: Path) -> dict:
    v = re.search(r"v(\d+)", stan.stem)
    v = int(v.group(1)) if v else 1
    if v >= 13:
        return {"v13": True}
    return {"v12": True} if v >= 12 else ({"v11": True} if v >= 11 else {})


def cover(draws, true) -> float:
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    return float(((true >= lo) & (true <= hi)).mean())


def main(stan: Path, laps: Path, seed: int, warmup: int, samples: int, report: Path | None) -> dict:
    des = build_design(load_push_laps(str(laps)))
    rng = np.random.default_rng(seed)
    flags, truth_fixed = flags_for(stan), dict(TRUTH)
    dt_bar = float(np.mean(des.dt[1:]))
    if flags.get("v13"):              # same true development / track fit, expressed the v1.3 way
        truth_fixed.update(sigma_car_event=float(np.sqrt(0.06 ** 2 * dt_bar + 0.30 ** 2)),
                           rho_dev=float(0.06 ** 2 * dt_bar / (0.06 ** 2 * dt_bar + 0.30 ** 2)))
    else:
        truth_fixed.update(sigma_dev=0.06, sigma_nu=0.30)
    truth = draw_params(des, rng, **flags, **truth_fixed)
    y = simulate_laps(des, truth, rng)
    fit = CmdStanModel(stan_file=str(stan)).sample(
        data=des.stan_data(y), chains=4, parallel_chains=4, iter_warmup=warmup, iter_sampling=samples,
        adapt_delta=0.95, inits=0.5, seed=seed, show_progress=False)
    sv = lambda k: fit.stan_variable(k)
    res = {"stan": str(stan),
           "coverage": {"driver": cover(sv("delta"), truth["delta"]), "car_trend": cover(sv("theta"), truth["theta"]),
                        "car_x_track": cover(sv("nu"), truth["nu"]), "track": cover(sv("tau")[:, :, 1:], truth["tau"][:, 1:])},
           "hyper": {}}
    if "gamma" in truth:
        res["coverage"]["gamma"] = cover(sv("gamma"), truth["gamma"])
    for k in ["sigma_dev", "sigma_nu", "sigma_driver", "pi_bad", "loss_clean", "sigma"]:
        lo, med, hi = np.percentile(sv(k), [5, 50, 95])
        res["hyper"][k] = {"truth": float(truth[k]), "median": float(med), "lo": float(lo), "hi": float(hi),
                           "inside": bool(lo <= truth[k] <= hi)}
    print(json.dumps(res, indent=1))
    if report:
        Path(report).write_text(json.dumps(res, indent=1))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        lines = [f"## Recovery: `{stan.name}`", "| component | 90% coverage |", "|---|---|"]
        lines += [f"| {k} | {v:.0%} |" for k, v in res["coverage"].items()]
        lines += ["", "| hyperparameter | truth | posterior median [90%] | inside |", "|---|---|---|---|"]
        lines += [f"| `{k}` | {h['truth']:.3f} | {h['median']:.3f} [{h['lo']:.3f}, {h['hi']:.3f}] | {'✅' if h['inside'] else '❌'} |"
                  for k, h in res["hyper"].items()]
        Path(summary).open("a").write("\n".join(lines) + "\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stan", type=Path, required=True)
    ap.add_argument("--laps", type=Path, default=Path("data/raw/quali_laps_2026.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--report", type=Path, default=None)
    a, _ = ap.parse_known_args()
    main(a.stan, a.laps, a.seed, a.warmup, a.samples, a.report)
