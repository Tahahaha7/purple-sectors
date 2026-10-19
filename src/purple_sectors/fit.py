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


def ess(fit, name) -> float:
    """Effective sample size of a scalar (autocorrelation with Geyer's initial positive sequence)."""
    x = fit.draws_pd(vars=[name])[name].to_numpy().reshape(fit.chains, -1)
    m, n = x.shape
    c = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(c, 2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n].mean(axis=0) / n
    w = x.var(axis=1, ddof=1).mean(); b = n * x.mean(axis=1).var(ddof=1)
    var = (n - 1) / n * w + b / n
    rho = 1 - (w - acov * n / (n - 1)) / var if var > 0 else np.ones(n)
    rho[0], s, t = 1.0, 0.0, 0
    while t + 1 < n and rho[t] + rho[t + 1] > 0:
        s += rho[t] + rho[t + 1]; t += 2
    return float(m * n / max(1e-9, 2 * s - 1))


def summary_line(fit, name) -> str:
    lo, med, hi = np.percentile(fit.stan_variable(name), [5, 50, 95])
    return f"| `{name}` | {med:.3f} | [{lo:.3f}, {hi:.3f}] |"


def gate(div: int, total: int, rhat: float, max_div_frac: float = 0.002, max_r: float = 1.01,
         warn_div_frac: float = 0.01, warn_r: float = 1.05) -> str:
    """Three-level quality gate: 'pass' publishes, 'warn' publishes with a flag, 'fail' blocks."""
    if div <= max_div_frac * total and rhat <= max_r:
        return "pass"
    if div <= warn_div_frac * total and rhat <= warn_r:
        return "warn"
    return "fail"


def main(laps: Path, stan: Path, out: Path, warmup: int, samples: int, seed: int,
         max_div_frac: float, max_r: float, max_round: int | None = None, allow_fail: bool = False,
         warn_div_frac: float = 0.01, warn_r: float = 1.05, save_loglik: bool = False,
         report: Path | None = None) -> int:
    des = build_design(load_push_laps(str(laps), max_round=max_round))
    print(f"{des.N} push laps · {des.D} drivers · {des.C} cars · {des.E} events")
    fit = CmdStanModel(stan_file=str(stan)).sample(
        data=des.stan_data(des.laps["y"].to_numpy()), chains=4, parallel_chains=4,
        iter_warmup=warmup, iter_sampling=samples, adapt_delta=0.95, inits=0.5,
        seed=seed, show_progress=False)

    names = set(getattr(getattr(fit, "metadata", None), "stan_vars", None) or {})   # what this model actually has
    hyper = [v for v in HYPER + ["sigma_car_event", "rho_dev"] if not names or v in names]
    mv = fit.method_variables()
    div = int(mv["divergent__"].sum())
    total = int(mv["divergent__"].size)
    rhat, worst = max_rhat(fit, hyper + LATENT)
    status = gate(div, total, rhat, max_div_frac, max_r, warn_div_frac, warn_r)
    print(f"divergences {div}/{total} · max R-hat {rhat:.3f} ({worst}) · {status.upper()}")

    out_vars = {v: fit.stan_variable(v).astype("float32") for v in hyper + LATENT + DERIVED}
    out_vars["p_bad_mean"] = fit.stan_variable("p_bad").mean(axis=0).astype("float32")
    if save_loglik:                                   # for LOO comparison between model versions
        out_vars["log_lik"] = fit.stan_variable("log_lik")[::4].astype("float32")
    np.savez_compressed(out, **out_vars, divergent=mv["divergent__"], treedepth=mv["treedepth__"],
                        drivers=np.array(des.drivers), cars=np.array(des.cars),
                        events=np.array(des.events), y=des.laps["y"].to_numpy(),
                        driver=des.driver, car=des.car, event=des.event, seg=des.seg,
                        divergences=div, draws_total=total, max_rhat=rhat, worst=worst, passed=status == "pass", status=status)
    print(f"saved {out}")
    if report:
        import json
        rep = {"stan": str(stan), "max_round": max_round, "laps": des.N, "events": des.E,
               "divergences": div, "draws": total, "max_rhat": rhat, "worst": worst, "status": status,
               "ess": {v: round(ess(fit, v)) for v in ["sigma_dev", "sigma_nu", "sigma_driver", "pi_bad"] if v in hyper},
               "car_share": float(np.median(fit.stan_variable("car_share").mean(axis=1))),
               "medians": {v: float(np.median(fit.stan_variable(v))) for v in ["sigma_dev", "sigma_nu", "sigma_driver", "pi_bad", "loss_clean"] if v in hyper}}
        Path(report).write_text(json.dumps(rep, indent=1))

    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):     # shows on the Actions run page
        cs = np.percentile(fit.stan_variable("car_share").mean(axis=1), [5, 50, 95])
        badge = {"pass": "✅ passed", "warn": "⚠️ passed with a warning", "fail": "❌ failed"}[status]
        lines = [f"## Model fit: {badge}",
                 f"{des.N} push laps · {des.E} events · divergences {div}/{total} · max R-hat {rhat:.3f} (`{worst}`)",
                 f"\n**Car share:** {cs[1]:.3f} [{cs[0]:.3f}, {cs[2]:.3f}]\n",
                 "| parameter | median | 90% |", "|---|---|---|"]
        lines += [summary_line(fit, v) for v in ["sigma_driver", "pi_bad", "loss_clean", "mu_gamma"]]
        Path(summary).open("a").write("\n".join(lines) + "\n")
    return 0 if status != "fail" or allow_fail else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--laps", type=Path, default=Path("data/raw/quali_laps_2026.csv"))
    ap.add_argument("--stan", type=Path, default=Path("stan/quali_v12.stan"))
    ap.add_argument("--out", type=Path, default=Path("posterior_v12.npz"))
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-div-frac", type=float, default=0.002, help="pass: max share of divergent draws")
    ap.add_argument("--max-rhat", type=float, default=1.01, help="pass: max R-hat")
    ap.add_argument("--warn-div-frac", type=float, default=0.01, help="warn (publish with a flag): divergences")
    ap.add_argument("--warn-rhat", type=float, default=1.05, help="warn (publish with a flag): R-hat")
    ap.add_argument("--max-round", type=int, default=None, help="fit only rounds 1..N (history snapshots)")
    ap.add_argument("--allow-fail", action="store_true", help="exit 0 even if diagnostics fail (backfill)")
    ap.add_argument("--save-loglik", action="store_true", help="store log_lik draws (model comparison)")
    ap.add_argument("--report", type=Path, default=None, help="write diagnostics as JSON here")
    a, _ = ap.parse_known_args()
    sys.exit(main(a.laps, a.stan, a.out, a.warmup, a.samples, a.seed, a.max_div_frac, a.max_rhat,
                  a.max_round, a.allow_fail, a.warn_div_frac, a.warn_rhat, a.save_loglik, a.report))
