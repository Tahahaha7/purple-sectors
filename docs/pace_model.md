# The pace model

One model for every qualifying push lap since 2023. It answers three questions with one set
of numbers:

1. **Which car is quickest, driver taken out.**
2. **Which driver is quickest, car taken out.**
3. **Where would driver X qualify in car Y** (the counterfactual grid on the Soft page).

Everything is in % of lap time; 1% is about 0.9 s on a 90-second lap.

## The model in one line

```
lap = baseline(session, segment) + evolution(session) × time + car + driver + fit(driver, car) + tyres + execution
```

| term | what it is | how it is identified |
|---|---|---|
| **car** | a team in one season; level + random walk over the season's rounds (in √weeks), which is the persistent pace (`trend`), plus a one-off deviation each round (`car`); both centred on the field at every round | two drivers in the same car, every lap |
| **driver** | skill relative to that season's grid; random walk across seasons, starting from a rookie offset, with extra movement in seasons two and three | teammates (within a car) and team moves (across cars) |
| **fit** | one shrunk term per (driver, car) pair that happened | drivers who drove more than one car tell us how big fit effects are; pairs that never happened get the prior |
| **execution** | every lap loses some time to the ideal: a clean ex-Gaussian or, with probability π, a compromised one that loses more | the shape of the lap-time distribution |
| tyres, evolution | medium vs soft, laps on the set, a per-session slope within a segment | covariates |

What this buys over the previous two-stage setup (history → marginal priors → 2026):

- One posterior, so all the correlations between drivers, cars and seasons are kept; the
  previous export dropped them, which made teammate-gap priors too wide and cross-team sums
  too narrow.
- Driver and car are defined on the same quantity (every push lap) instead of best laps for
  history and all laps for 2026.
- The `fit` term is what makes the three claims honest: a driver who never changed team has
  skill and fit entangled, and the interval says so. A driver who moved pins both down.
- Rookies get a rookie offset and a directional improvement in years two and three instead of
  "average driver, more variance".
- Deleted laps (track limits) are kept: the car did go that fast. Use `--drop-deleted` to compare.

Car rankings and the counterfactual grid use the persistent pace (`trend`) at the latest
round; `last` in the JSON is what the car did at that particular track, deviation included.

## What the counterfactual grid is, and is not

For each shown driver and each current car: `trend(now) + driver(now) + fit(driver, car)`,
compared with the other drivers in their actual seats, so the number is "where would they
qualify today, other things equal". When the pair has been seen, `fit` is its posterior; when
not, it is a draw from the prior `N(0, σ_fit)`, so the interval widens and the square is drawn
dashed. It is a prediction under the model's additive structure, not a guarantee; a car built
around one driver, a mid-season upgrade for one side of the garage, or a driver who simply
suits a car are all things `fit` can only partly absorb.

Drivers outside the main teammate network (see `connectivity` in the JSON) are only comparable
with their own teammates. The notebook prints the components.

## Validation

- **Recovery**: `python -m purple_sectors.pace.fit --recovery` simulates a history on the real
  design from a known truth and reports how often 90% intervals contain it (want 80–95% for
  driver, car, fit and the hyperparameters).
- **Quality gate** on every fit: zero divergences, R-hat ≤ 1.01, bulk and tail ESS ≥ 400,
  E-BFMI ≥ 0.3 and no saturated tree depths to pass; a narrow warn band; `--retry` refits
  once more carefully before anything is published.
- **Forecast scorecard** (live on Soft and Hard): every round is forecast before it happens and
  scored after (favourite right, Brier skill for Q3, rank correlation).
- **Still to do**: rolling-origin hold-out (fit ≤ 2023 → predict 2024, ≤ 2024 → 2025,
  ≤ 2025 → 2026) scoring sign, Brier/log score and coverage of teammate gaps, rookies included;
  best-lap calibration of head-to-head probabilities.

## Running it

```
python -m purple_sectors.fetch --year 2023      # and 2024, 2025, 2026
python -m purple_sectors.pace.fit --out posterior_pace.npz --report fit.json --retry
python -m purple_sectors.pace.export --posterior posterior_pace.npz --laps data/raw/quali_laps_2026.csv \
    --out site/data --snapshot-dir site/data/snapshots
pytest                                            # a few seconds, no CmdStan or network
```

`notebooks/01_pace_model.ipynb` does all of this in Colab, the same way as the earlier notebooks.

## What changed from the previous model (October 2026)

The site used to run two models: a history model on 2022–25 best laps whose per-driver summary
became the prior of a 2026 lap model (v2.0; before that v1 → v1.3 added within-segment track
evolution, the clean/compromised mixture and the total-plus-share car parametrisation). Keeping
what those versions taught and dropping the hand-off:

- **One fit, 2023 onward, lap level.** The two-stage export passed only each driver's marginal
  mean and sd, losing the strong posterior correlations between teammates and across the
  network; and it put a best-lap prior on an all-laps parameter. One posterior removes both.
- **Driver-by-car fit.** Previously the car and driver were strictly additive, which over-credits
  a driver whose teammates all struggle in the same car. `fit` absorbs that, and it makes the
  counterfactual grid honest for pairs that never happened.
- **Rookies** start from an estimated debut offset with a directional improvement in years two
  and three, instead of "average driver, wider variance".
- **Deleted laps are kept** (track limits do not slow a car down); **substitutes stay in the
  model** and are only filtered on the pages.
- **Quality gate** now checks ESS, E-BFMI and tree depth, and refits once before publishing.
- The **best-lap calibration** problem (head-to-head and pole probabilities about 20% too
  cautious) is unchanged and still open, as is a rolling-origin hold-out.
