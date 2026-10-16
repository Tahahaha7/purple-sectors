# purple-sectors

[![Update model](https://github.com/Tahahaha7/purple-sectors/actions/workflows/update.yml/badge.svg)](https://github.com/Tahahaha7/purple-sectors/actions/workflows/update.yml)
[![Deploy site](https://github.com/Tahahaha7/purple-sectors/actions/workflows/pages.yml/badge.svg)](https://github.com/Tahahaha7/purple-sectors/actions/workflows/pages.yml)

A Bayesian model of Formula 1 qualifying, refit automatically after every session of the 2026 season.
Every push lap is decomposed into **car**, **driver**, **track evolution** and **execution**, with full uncertainty.

**Live site: [tahahaha7.github.io/purple-sectors](https://tahahaha7.github.io/purple-sectors/)**

| Page | For | What it shows |
|---|---|---|
| [**[S] Soft**](https://tahahaha7.github.io/purple-sectors/) | Every fan | Quickest cars, teammate battles as dot plots, what changed this weekend, season movers |
| [**[M] Medium**](https://tahahaha7.github.io/purple-sectors/medium.html) | Enthusiasts | Factor graph of the model, animated round by round; teammate gaps as distributions that narrow over the season |
| [**[H] Hard**](https://tahahaha7.github.io/purple-sectors/hard.html) | Statisticians | Equations, priors, inference, validation, limitations, and the live current fit |
| [**[I] Inter**](https://tahahaha7.github.io/purple-sectors/inter.html) | The curious | How each session moves the model: evidence particles and joint posterior clouds |

## Headline findings (2026, 17 rounds)
- **Cars span about 2.5 s a lap; teammates differ by about 0.14 s.** The car sets the order, the driver settles the garage.
- **About 1 in 10 push laps is compromised** (traffic, mistakes, yellows) and costs about 1.1 s; a clean lap is about 0.25 s off the limit.
- Drivers are compared **with their teammate only**; cross-team rankings need a multi-season model (planned).

## Model (production: `stan/quali_v12.stan`)
`y = 100·log(lap time)`, so every parameter is in % of lap time.
```
y ~ (1 − π)·ExGauss(μ, σ, 1/loss_clean) + π·ExGauss(μ, σ, 1/loss_bad)
μ = event + segment step + within-segment evolution · t
  + car trend (random walk in √weeks) + car × track fit + driver + tyres
```
Full details on the [Hard page](https://tahahaha7.github.io/purple-sectors/hard.html); development record in [`docs/model_log.md`](docs/model_log.md).

## Pipeline
1. **Every Sunday 06:00 UTC** (retry Monday), `update.yml`: fetch new sessions from [OpenF1](https://openf1.org) → fit v1.2 in Stan → quality gate → export site data and a per-round snapshot → commit → redeploy.
2. **Quality gate:** pass (R-hat ≤ 1.01, divergences ≤ 0.2%) publishes; warn (≤ 1.05, ≤ 1%) publishes with a flag; fail blocks.
3. **History:** `backfill.yml` refits rounds 1..k for every k in parallel, so Medium and Inter can replay the season.

## Repository
```
src/purple_sectors/fetch.py      OpenF1 → data/raw/quali_laps_<year>.csv (rate-limited, cached)
src/purple_sectors/prepare.py    push-lap filter, one car per driver, design
src/purple_sectors/fit.py        CmdStan fit + three-level quality gate
src/purple_sectors/export.py     posterior → site JSON, per-round snapshots and joint draws
src/purple_sectors/simulate.py   NumPy mirror of the Stan models (prior checks, recovery)
stan/                            v1, v1-B (sensitivity), v1.1, v1.2 (production)
notebooks/                       recovery and model-comparison notebooks (Colab)
site/                            the static website (GitHub Pages)
.github/workflows/               update, backfill, pages
```

## Reproduce
```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m purple_sectors.fetch --year 2026
python -m purple_sectors.fit --stan stan/quali_v12.stan --out posterior_v12.npz
python -m purple_sectors.export --posterior posterior_v12.npz --out site/data --snapshot-dir site/data/snapshots
```

## Data and licence
Lap timing from [OpenF1](https://openf1.org). Code under the MIT licence.
An independent fan project, not associated with Formula 1.
