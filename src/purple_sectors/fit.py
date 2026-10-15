"""Fit the production model on the latest data and save a thinned posterior (.npz).

Exits with status 1 if the sampler diagnostics fail, so the scheduled pipeline
never publishes a bad fit.

Usage:  python -m purple_sectors.fit --laps data/raw/quali_laps_2026.csv \
            --stan stan/quali_v12.stan --out posterior_v12.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel

from purple_sectors.prepare import build_design, load_push_laps

HYPER = ["m", "sigma_kappa", "mu_gamma", "sigma_gamma", "sigma_car", "sigma_dev", "sigma_nu",
         "sigma_driver", "beta_m", "beta_a", "sigma", "pi_bad", "loss_clean", "loss_gap"]
LATENT = ["gamma", "delta", "theta", "nu", "tau", "alpha"]
DERIVED = ["car_share", "loss_bad"]


def max_rhat(fit, variables) -> tuple[float, str]:
    """Split R-hat (Gelman et al.): the maximum over all scalar components, and which one it is."""
    df = fit.draws_pd(vars=variables)
    cols = [c for c in df.columns if not c.endswith("__")]
    n = fit.num_draws_sampling
    x = df[cols].to_numpy().reshape(fit.chains, n, len(cols)).transpose(1, 0, 2)
    h = n // 2
    x = np.concatenate([x[:h], x[h:2 * h]], axis=1)
    w = x.var(axis=0, ddof=1).mean(axis=0)
    b = h * x.mean(axis=0).var(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.sqrt(((h - 1) / h * w + b / h) / w)
    k = int(np.nanargmax(r))
    return float(r[k]), cols[k]


def summary_line(fit, name) -> str:
    lo, med, hi = np.percentile(fit.stan_variable(name), [5, 50, 95])
    return f"| `{name}` | {med:.3f} | [{lo:.3f}, {hi:.3f}] |"


def main(laps: Path, stan: Path, out: Path, warmup: int, samples: int, seed: int,
         max_div_frac: float, max_r: float, max_round: int | None = None, allow_fail: bool = False) -> int:
    des = build_design(load_push_laps(str(laps), max_round=max_round))
    print(f"{des.N} push laps · {des.D} drivers · {des.C} cars · {des.E} events")
    fit = CmdStanModel(stan_file=str(stan)).sample(
        data=des.stan_data(des.laps["y"].to_numpy()), chains=4, parallel_chains=4,
        iter_warmup=warmup, iter_sampling=samples, adapt_delta=0.95, inits=0.5,
        seed=seed, show_progress=False)

    mv = fit.method_variables()
    div = int(mv["divergent__"].sum())
    total = int(mv["divergent__"].size)
    rhat, worst = max_rhat(fit, HYPER + LATENT)
    ok = div <= max_div_frac * total and rhat <= max_r
    print(f"divergences {div}/{total} · max R-hat {rhat:.3f} ({worst}) · {'OK' if ok else 'FAILED'}")

    out_vars = {v: fit.stan_variable(v).astype("float32") for v in HYPER + LATENT + DERIVED}
    out_vars["p_bad_mean"] = fit.stan_variable("p_bad").mean(axis=0).astype("float32")
    np.savez_compressed(out, **out_vars, divergent=mv["divergent__"], treedepth=mv["treedepth__"],
                        drivers=np.array(des.drivers), cars=np.array(des.cars),
                        events=np.array(des.events), y=des.laps["y"].to_numpy(),
                        driver=des.driver, car=des.car, event=des.event, seg=des.seg,
                        divergences=div, draws_total=total, max_rhat=rhat, worst=worst, passed=ok)
    print(f"saved {out}")

    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):     # shows on the Actions run page
        cs = np.percentile(fit.stan_variable("car_share").mean(axis=1), [5, 50, 95])
        lines = [f"## Model fit: {'✅ passed' if ok else '❌ failed'}",
                 f"{des.N} push laps · {des.E} events · divergences {div}/{total} · max R-hat {rhat:.3f} (`{worst}`)",
                 f"\n**Car share:** {cs[1]:.3f} [{cs[0]:.3f}, {cs[2]:.3f}]\n",
                 "| parameter | median | 90% |", "|---|---|---|"]
        lines += [summary_line(fit, v) for v in ["sigma_driver", "pi_bad", "loss_clean", "mu_gamma"]]
        Path(summary).open("a").write("\n".join(lines) + "\n")
    return 0 if ok or allow_fail else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--laps", type=Path, default=Path("data/raw/quali_laps_2026.csv"))
    ap.add_argument("--stan", type=Path, default=Path("stan/quali_v12.stan"))
    ap.add_argument("--out", type=Path, default=Path("posterior_v12.npz"))
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-div-frac", type=float, default=0.002, help="max share of divergent draws")
    ap.add_argument("--max-rhat", type=float, default=1.01)
    ap.add_argument("--max-round", type=int, default=None, help="fit only rounds 1..N (history snapshots)")
    ap.add_argument("--allow-fail", action="store_true", help="exit 0 even if diagnostics fail (backfill)")
    a, _ = ap.parse_known_args()
    sys.exit(main(a.laps, a.stan, a.out, a.warmup, a.samples, a.seed, a.max_div_frac, a.max_rhat,
                  a.max_round, a.allow_fail))
