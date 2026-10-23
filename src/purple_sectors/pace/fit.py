"""Fit stan/pace.stan, check the sampler, save the posterior.

Usage:
  python -m purple_sectors.pace.fit --years 2023 2024 2025 2026 --out posterior_pace.npz --report fit.json
  python -m purple_sectors.pace.fit --recovery --report recovery.json   # fake laps on the real design
  python -m purple_sectors.pace.fit --prior-only --out prior.npz

Exit code 0 when the quality gate passes or warns, 1 when it fails (unless --allow-fail).
The gate looks at divergences, R-hat, bulk and tail ESS, E-BFMI and tree depth; a warn is
published with a flag, and with --retry a fail is refit once with a more careful sampler first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len
from scipy.stats import norm, rankdata

from . import simulate as sim
from .data import connectivity, design_from_files

LATENT = ["alpha", "gamma", "car", "trend", "skill", "fit"]
DERIVED = ["sigma_dev", "sigma_nu", "loss_bad"]


# ----------------------------------------------------------------------------- diagnostics

def _split(x):
    """(iterations, chains, params) -> (iterations/2, 2*chains, params)."""
    n = x.shape[0] // 2
    return np.concatenate([x[:n], x[n:2 * n]], axis=1)


def _rank_normal(x):
    n, m, k = x.shape
    flat = x.reshape(n * m, k)
    ranks = rankdata(flat, axis=0)
    return norm.ppf((ranks - 0.375) / (n * m + 0.25)).reshape(n, m, k)


def _rhat(z):
    n = z.shape[0]
    w = z.var(axis=0, ddof=1).mean(axis=0)
    b = n * z.mean(axis=0).var(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(((n - 1) / n * w + b / n) / w)


def _ess(z):
    """Effective sample size (Geyer's initial monotone sequence), vectorised over params."""
    n, m, k = z.shape
    zc = z - z.mean(axis=0, keepdims=True)
    nfft = next_fast_len(2 * n)
    f = np.fft.rfft(zc, n=nfft, axis=0)
    acov = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[:n] / n          # (n, m, k)
    chain_var = acov[0] * n / (n - 1)
    w = chain_var.mean(axis=0)
    var_plus = w * (n - 1) / n + n * z.mean(axis=0).var(axis=0, ddof=1) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = 1 - (w - acov.mean(axis=1)) / var_plus                     # (n, k)
    pairs = rho[: (n // 2) * 2].reshape(n // 2, 2, k).sum(axis=1)       # (n/2, k)
    negative = np.cumsum(pairs < 0, axis=0) > 0                          # stop at the first negative pair
    pairs = np.where(negative, 0, pairs)
    pairs = np.minimum.accumulate(np.where(negative, 0, pairs), axis=0)  # initial monotone sequence
    tau = -1 + 2 * pairs.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.minimum(n * m / tau, n * m * np.log10(n * m))


def diagnostics(fit, max_treedepth: int) -> dict:
    """Divergences, tree depth, E-BFMI, and R-hat / ESS over every non-constant parameter."""
    cols = list(fit.column_names)
    keep = [i for i, c in enumerate(cols) if not c.endswith("__") and not c.startswith("log_lik")]
    draws = np.asarray(fit.draws(inc_warmup=False), dtype=float)[:, :, keep]   # (iter, chain, k)
    names = [cols[i] for i in keep]
    varying = draws.std(axis=(0, 1)) > 1e-12
    x = _split(draws[:, :, varying])
    z = _rank_normal(x)
    rhat = np.fmax(_rhat(z), _rhat(_rank_normal(np.abs(x - np.median(x, axis=(0, 1), keepdims=True)))))
    ess_bulk = _ess(z)
    q05, q95 = np.quantile(x, [0.05, 0.95], axis=(0, 1))
    ess_tail = np.fmin(_ess(_rank_normal((x <= q05).astype(float))), _ess(_rank_normal((x <= q95).astype(float))))
    vnames = [nm for nm, v in zip(names, varying) if v]
    mv = fit.method_variables()
    div = int(mv["divergent__"].sum())
    total = int(mv["divergent__"].size)
    energy = mv["energy__"]
    bfmi = (np.diff(energy, axis=0) ** 2).sum(axis=0) / ((energy - energy.mean(axis=0)) ** 2).sum(axis=0)
    worst = int(np.nanargmax(rhat))
    return dict(divergences=div, draws=total, div_frac=div / total,
                max_rhat=float(np.nanmax(rhat)), worst=vnames[worst],
                min_ess_bulk=float(np.nanmin(ess_bulk)), min_ess_tail=float(np.nanmin(ess_tail)),
                min_bfmi=float(np.min(bfmi)),
                treedepth_frac=float((mv["treedepth__"] >= max_treedepth).mean()))


def gate(d: dict) -> str:
    if (d["divergences"] == 0 and d["max_rhat"] <= 1.01 and d["min_ess_bulk"] >= 400
            and d["min_ess_tail"] >= 400 and d["min_bfmi"] >= 0.3 and d["treedepth_frac"] == 0):
        return "pass"
    if (d["div_frac"] <= 0.001 and d["max_rhat"] <= 1.02 and d["min_ess_bulk"] >= 200
            and d["min_ess_tail"] >= 200 and d["min_bfmi"] >= 0.2 and d["treedepth_frac"] <= 0.01):
        return "warn"
    return "fail"


# ----------------------------------------------------------------------------- fitting

def sample(stan: Path, data: dict, warmup: int, samples: int, seed: int, adapt_delta: float,
           max_treedepth: int, chains: int = 4):
    from cmdstanpy import CmdStanModel
    return CmdStanModel(stan_file=str(stan)).sample(
        data=data, chains=chains, parallel_chains=chains, iter_warmup=warmup, iter_sampling=samples,
        adapt_delta=adapt_delta, max_treedepth=max_treedepth, inits=0.5, seed=seed, show_progress=False)


def posterior_dict(fit, des, keep: int, seed: int) -> dict:
    """Everything export needs, as plain arrays (no pickles)."""
    total = fit.num_draws_sampling * fit.chains
    step = max(1, total // keep)
    out = {}
    for v in sim.HYPERS + DERIVED + LATENT:
        out[v] = fit.stan_variable(v)[::step].astype("float32")
    rng = np.random.default_rng(seed)
    pick = rng.choice(out["skill"].shape[0], min(200, out["skill"].shape[0]), replace=False)
    pb = np.zeros(des.N)
    for i in pick:
        p = sim.params_from_draw(out, i)
        pb += sim.p_bad(des.y, sim.mu_of(des, p), p["sigma"], p["loss_clean"], p["loss_bad"], p["pi_bad"])
    out["p_bad_mean"] = (pb / len(pick)).astype("float32")
    ev = des.events
    meta = dict(
        seasons=des.seasons, teams={str(s): des.teams[s] for s in des.seasons}, drivers=des.drivers,
        events=[dict(season=int(s), round=int(r), r=int(k), e=int(e), date=str(d)[:10],
                     name=str(ev["event"].iloc[i]) if "event" in ev else "",
                     location=str(ev["location"].iloc[i]) if "location" in ev else "")
                for i, (s, r, k, e, d) in enumerate(zip(ev["season"], ev["round"], ev["r"], ev["e"], ev["date"]))],
        pairs=[list(map(int, pr)) for pr in des.pairs], bases=[list(b) for b in des.bases],
        debuts=des.debuts, present=des.present.tolist(), rookie=des.rookie.tolist(), young=des.young.tolist(),
        names=driver_names(des), connectivity=connectivity(des), best_laps=des.N2)
    out["meta_json"] = np.array(json.dumps(meta))
    for k in ("season", "rnd", "team", "driver", "pair", "base", "event"):
        out[k] = getattr(des, k).astype("int32")
    for k in ("t", "medium", "tyre_age", "y", "y_ref"):
        out[k] = getattr(des, k).astype("float32")
    out["dt"] = des.dt.astype("float32")
    return out


def driver_names(des) -> dict:
    """Display name and number per driver code, when the lap file carries them."""
    import pandas as pd
    laps = des.laps
    out = {}
    frames = [laps] + ([des.best] if des.best is not None else [])
    for d in des.drivers:
        rows = next((f[f["driver"] == d] for f in frames if (f["driver"] == d).any()), None)
        if rows is None:
            out[d] = dict(name=d, number=0, team="", season=0)
            continue
        last = rows.iloc[-1]
        name_col = next((c for c in ("full_name", "name") if c in rows), None)
        num_col = next((c for c in ("driver_number", "number") if c in rows), None)
        number = pd.to_numeric(rows[num_col], errors="coerce").dropna() if num_col else pd.Series(dtype=float)
        out[d] = dict(name=str(last[name_col]) if name_col else d,
                      number=int(number.iloc[-1]) if len(number) else 0,
                      team=str(last["team"]), season=int(last["season"]))
    return out


def coverage(draws, truth, mask=None) -> float:
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    inside = (truth >= lo) & (truth <= hi)
    return float(inside[mask].mean() if mask is not None else inside.mean())


def run(a) -> dict:
    des = design_from_files(a.laps_dir, a.years, a.debuts, a.max_round, a.history, a.history_seasons,
                            push_threshold=a.push_threshold, keep_deleted=not a.drop_deleted)
    net = connectivity(des)
    print(f"{des.N} push laps + {des.N2} earlier best laps · {des.D} drivers · {sum(des.T)} cars · {des.E} events"
          f" · {des.P} driver-car pairs · connected: {net['connected']}")
    if not net["connected"]:
        print("not connected; drivers outside the main group:", net["components"][1:])
    rng = np.random.default_rng(a.seed)
    truth = None
    y, y2 = des.y, des.y2
    if a.recovery:
        truth = sim.draw_params(des, rng, **sim.TRUTH)
        y = sim.simulate_laps(des, truth, rng)
        y2 = sim.simulate_best(des, truth, rng)
    data = des.stan_data(y, y2, prior_only=a.prior_only, save_loglik=a.save_loglik)

    t0 = time.time()
    fit = sample(a.stan, data, a.warmup, a.samples, a.seed, a.adapt_delta, a.max_treedepth)
    diag = diagnostics(fit, a.max_treedepth)
    status = gate(diag)
    print(f"quality gate: {status} · {json.dumps(diag, default=float)}")
    if status == "fail" and a.retry:
        print("refitting once with adapt_delta 0.99 and a deeper tree")
        fit = sample(a.stan, data, a.warmup, a.samples, a.seed + 1, 0.99, a.max_treedepth + 1)
        diag = diagnostics(fit, a.max_treedepth + 1)
        status = gate(diag)
        print(f"quality gate after retry: {status} · {json.dumps(diag, default=float)}")
    diag["status"] = status
    diag["minutes"] = round((time.time() - t0) / 60, 1)

    rep = dict(laps=des.N, best_laps=des.N2, drivers=des.D, cars=int(sum(des.T)), events=des.E, pairs=des.P,
               seasons=des.seasons, connected=net["connected"], components=len(net["components"]),
               diagnostics=diag, status=status,
               medians={k: round(float(np.median(fit.stan_variable(k))), 4) for k in sim.HYPERS})
    if truth is not None:
        used = np.zeros((des.S, des.TMAX, des.RMAX), dtype=bool)
        for s, (T, R) in enumerate(zip(des.T, des.R)):
            used[s, :T, :R] = True
        hyp = {k: dict(truth=truth[k], median=rep["medians"][k],
                       inside=bool(np.percentile(fit.stan_variable(k), 5) <= truth[k] <= np.percentile(fit.stan_variable(k), 95)))
               for k in sim.HYPERS}
        rep["recovery"] = dict(
            skill=coverage(fit.stan_variable("skill"), truth["skill"], des.present == 1),
            car=coverage(fit.stan_variable("car"), truth["car"], used),
            fit=coverage(fit.stan_variable("fit"), truth["fit"]),
            hypers_inside=sum(v["inside"] for v in hyp.values()) / len(hyp), hypers=hyp)
    if a.report:                                   # written first, so a failure below cannot lose it
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "medians"}, indent=1, default=float))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(summary).open("a").write(step_summary(rep))

    if a.out:
        post = posterior_dict(fit, des, a.keep, a.seed)
        if a.save_loglik:
            post["log_lik"] = fit.stan_variable("log_lik")[::4].astype("float32")
        post["diag_json"] = np.array(json.dumps(diag))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out, **post)
        print(f"wrote {a.out}")
    return rep


def step_summary(rep: dict) -> str:
    d = rep["diagnostics"]
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[rep["status"]]
    lines = [f"## pace model {icon} {rep['status']}",
             f"{rep['laps']} push laps + {rep.get('best_laps', 0)} earlier best laps · {rep['drivers']} drivers · {rep['cars']} cars · {rep['events']} events"
             f" · {rep['pairs']} driver-car pairs · connected: {rep['connected']}",
             f"divergences {d['divergences']}/{d['draws']} · max R-hat {d['max_rhat']:.3f} ({d['worst']})"
             f" · min ESS bulk {d['min_ess_bulk']:.0f} tail {d['min_ess_tail']:.0f}"
             f" · min E-BFMI {d['min_bfmi']:.2f} · {d['minutes']} min", "",
             "| parameter | median |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in rep["medians"].items()]
    if "recovery" in rep:
        r = rep["recovery"]
        lines += ["", f"**Recovery (90% intervals):** driver skill {r['skill']:.0%} · car {r['car']:.0%}"
                      f" · fit {r['fit']:.0%} · hyperparameters {r['hypers_inside']:.0%}"]
    return "\n".join(lines) + "\n"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    ap.add_argument("--laps-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--debuts", type=Path, default=Path("data/raw/driver_debuts.json"))
    ap.add_argument("--history", type=Path, default=Path("data/raw/history_quali_2022_2025.csv"),
                    help="Jolpica best laps; seasons before the first lap file are added as best-lap observations")
    ap.add_argument("--history-seasons", type=int, nargs="*", default=[2022])
    ap.add_argument("--stan", type=Path, default=Path("stan/pace.stan"))
    ap.add_argument("--out", type=Path, default=None, help="posterior .npz")
    ap.add_argument("--report", type=Path, default=None, help="diagnostics and summary as JSON")
    ap.add_argument("--max-round", type=int, default=None, help="latest season only up to this round")
    ap.add_argument("--push-threshold", type=float, default=7.0)
    ap.add_argument("--drop-deleted", action="store_true", help="exclude laps deleted for track limits")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--adapt-delta", type=float, default=0.95)
    ap.add_argument("--max-treedepth", type=int, default=10)
    ap.add_argument("--keep", type=int, default=2000, help="draws kept in the posterior file")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--retry", action="store_true", help="refit once, more carefully, if the gate fails")
    ap.add_argument("--recovery", action="store_true", help="fit simulated laps and check the truth is recovered")
    ap.add_argument("--prior-only", action="store_true")
    ap.add_argument("--save-loglik", action="store_true", help="keep lap log-likelihoods (for LOO)")
    ap.add_argument("--allow-fail", action="store_true", help="exit 0 even when the gate fails")
    return ap.parse_known_args(argv)[0]


if __name__ == "__main__":
    args = parse()
    rep = run(args)
    sys.exit(0 if rep["status"] != "fail" or args.allow_fail else 1)
