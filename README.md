# purple-sectors

A Bayesian model of Formula 1 qualifying pace. Every push lap of the 2026 season is decomposed into
**car**, **driver**, **track evolution**, and **execution**, with full uncertainty, and refit after each race weekend.

## Headline results (2026, rounds 1–16)
- **~98% of qualifying pace differences come from the car** (car share 0.983, 90% CI 0.964–0.991), robust to a neutral prior.
- **About 9% of push laps are compromised** (traffic, mistakes, yellows), costing ~1.4% each; a clean lap is ~0.28% (~250 ms) off a driver's potential.
- Driver skill spread is ~0.15% of lap time; v1 identifies drivers **relative to their teammate** only.

## Model (production: `stan/quali_v12.stan`)
`y = 100·log(lap time)`, so every parameter is in % of lap time.

```
y  ~ (1 − π)·ExGauss(μ, σ, 1/loss_clean) + π·ExGauss(μ, σ, 1/loss_bad)
μ  = event baseline + segment step + within-segment evolution·t
   + car trend (random walk in √weeks) + car × track fit + driver + tyres
```
Sum-to-zero constraints on car and driver effects; non-centred hierarchical priors; priors checked against
F1 plausibility limits (107% rule, typical teammate gaps). Full development history: [`docs/model_log.md`](docs/model_log.md).

## Repository
```
src/purple_sectors/fetch.py      OpenF1 → data/raw/quali_laps_<year>.csv (rate-limited, cached)
src/purple_sectors/prepare.py    push-lap filter, one car per driver, index design
src/purple_sectors/simulate.py   NumPy mirror of the Stan models (prior checks, recovery)
stan/quali.stan                  v1   · single ex-Gaussian, segment steps
stan/quali_share.stan            v1-B · neutral Beta(1,1) car-share prior (sensitivity)
stan/quali_v11.stan              v1.1 · + within-segment track evolution
stan/quali_v12.stan              v1.2 · + clean / compromised lap mixture (production)
notebooks/01–03                  recovery + real fits, run on Google Colab
```

## Reproduce
```bash
pip install -r requirements.txt
python src/purple_sectors/fetch.py --year 2026
```
Then open the notebooks in Colab (CmdStan is installed in the first cells).

## Data
Lap timing from the [OpenF1 API](https://openf1.org). Not affiliated with Formula 1.
