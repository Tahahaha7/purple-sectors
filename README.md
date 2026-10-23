# purple-sectors

[![Update model](https://github.com/Tahahaha7/purple-sectors/actions/workflows/update.yml/badge.svg)](https://github.com/Tahahaha7/purple-sectors/actions/workflows/update.yml) [![Deploy site](https://github.com/Tahahaha7/purple-sectors/actions/workflows/pages.yml/badge.svg)](https://github.com/Tahahaha7/purple-sectors/actions/workflows/pages.yml) [![CI](https://github.com/Tahahaha7/purple-sectors/actions/workflows/ci.yml/badge.svg)](https://github.com/Tahahaha7/purple-sectors/actions/workflows/ci.yml)

Who is quick because of the car, and who is quick anyway. A Bayesian model of Formula 1
qualifying, refit after every session.

**Live site: [tahahaha7.github.io/purple-sectors](https://tahahaha7.github.io/purple-sectors/)**

| Page | For | What it shows |
|---|---|---|
| **Soft** | fans | quickest drivers with the car removed, quickest cars with the driver removed, every driver in every car, the next qualifying forecast |
| **Medium** | enthusiasts | how the picture moved round by round |
| **Inter** | the curious | one driver at a time: ranking, where they would qualify in every car, their teammate battle |
| **Hard** | statisticians | the model, its priors, every check, and the live diagnostics |

## The model

Every push lap since 2023, in % of lap time (1% ≈ 0.9 s):

```
lap = baseline(session, segment) + evolution × time + car + driver + fit(driver, car) + tyres + execution
```

A car is a team in one season. Driver skill carries across seasons, so drivers who changed
team link every car to every other, and the car and driver parts can be told apart. `fit` is
a shrunk driver-by-car term: it is what lets the site put a driver in a car he never drove and
say how unsure that is. Details, limits and checks: [`docs/pace_model.md`](docs/pace_model.md).

## Pipeline

Every Sunday 06:00 UTC (retry Monday), `update.yml`: fetch new sessions from
[OpenF1](https://openf1.org) → fit `stan/pace.stan` on 2023–now → sampler checks (zero
divergences, R-hat ≤ 1.01, ESS ≥ 400, or one careful refit) → export the site's JSON and the
next-round forecast → commit → redeploy. `backfill.yml` refits rounds 1..k for every k, so
Medium and Inter can replay the season.

```
src/purple_sectors/fetch.py          OpenF1 → data/raw/quali_laps_<year>.csv (rate-limited, cached)
src/purple_sectors/history.py        Jolpica → driver debuts and the teammate network
src/purple_sectors/pace/data.py      laps → design
src/purple_sectors/pace/fit.py       CmdStan fit, checks, posterior file
src/purple_sectors/pace/export.py    posterior → site/data (soft, pace grid, snapshots)
src/purple_sectors/pace/forecast.py  next qualifying session and the scorecard
src/purple_sectors/pace/simulate.py  NumPy mirror of the Stan model (prior checks, recovery)
src/purple_sectors/rollover.py       end-of-season archive (docs/rollover.md)
stan/pace.stan                       the model
site/                                the static site (GitHub Pages)
notebooks/01_pace_model.ipynb        the whole pipeline in Colab, with the recovery check
tests/                               pytest, no network or CmdStan needed
```

## Reproduce

```
pip install -r requirements.txt
export PYTHONPATH=src
for y in 2023 2024 2025 2026; do python -m purple_sectors.fetch --year $y; done
python -m purple_sectors.pace.fit --years 2023 2024 2025 2026 --out posterior_pace.npz --report fit.json --retry
python -m purple_sectors.pace.export --posterior posterior_pace.npz --laps data/raw/quali_laps_2026.csv \
    --out site/data --snapshot-dir site/data/snapshots
```

```
pip install -r requirements-dev.txt
pytest                      # a few seconds
```

Lap timing from [OpenF1](https://openf1.org); debuts and teammate history from
[Jolpica](https://jolpi.ca). MIT licence. A fan project, not associated with Formula 1.
