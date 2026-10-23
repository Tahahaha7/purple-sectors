# purple-sectors

Who is quick because of the car, and who is quick anyway. A Bayesian model of Formula 1
qualifying, refit after every session.

**Live site: [tahahaha7.github.io/purple-sectors](https://tahahaha7.github.io/purple-sectors/)**

| Page | For | What it shows |
|---|---|---|
| **Soft** | fans | quickest cars with the driver taken out, quickest drivers with the car taken out, every driver in every car |
| **Medium** | enthusiasts | how the picture moved round by round |
| **Inter** | the curious | how one session's laps move the model |
| **Hard** | statisticians | the model, priors, checks and the live diagnostics |

## The model

Every push lap since 2023, in % of lap time (1% ≈ 0.9 s):

```
lap = baseline(session, segment) + evolution × time + car + driver + fit(driver, car) + tyres + execution
```

A car is a team in one season. Driver skill carries across seasons, so drivers who changed
team link every car to every other and the car and driver parts can be told apart. `fit` is
a shrunk driver-by-car term; it is what lets the site put a driver in a car he never drove and
say how unsure that is. Details, limits and checks: [`docs/pace_model.md`](docs/pace_model.md).

## Pipeline

Every Sunday 06:00 UTC: fetch new sessions from [OpenF1](https://openf1.org), fit
`stan/pace.stan`, run the sampler checks (zero divergences, R-hat ≤ 1.01, ESS ≥ 400, or a
careful refit), export `site/data/pace.json`, deploy.

```
src/purple_sectors/fetch.py        OpenF1 → data/raw/quali_laps_<year>.csv
src/purple_sectors/pace/data.py    laps → design
src/purple_sectors/pace/fit.py     fit, checks, posterior file
src/purple_sectors/pace/export.py  posterior → site JSON
src/purple_sectors/pace/simulate.py NumPy mirror of the Stan model (prior checks, recovery)
stan/pace.stan                     the model
site/                              the static site (GitHub Pages)
tests/                             pytest, no network or CmdStan needed
```

## Reproduce

```
pip install -r requirements.txt
export PYTHONPATH=src
for y in 2023 2024 2025 2026; do python -m purple_sectors.fetch --year $y; done
python -m purple_sectors.pace.fit --out posterior_pace.npz --report fit.json --retry
python -m purple_sectors.pace.export --posterior posterior_pace.npz --out site/data/pace.json
pytest
```

Lap timing from [OpenF1](https://openf1.org), driver debuts from [Jolpica](https://jolpi.ca).
MIT licence. A fan project, not associated with Formula 1.
