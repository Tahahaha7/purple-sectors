# Model development log

All fits: 4 chains × 3,000 draws, 2026 rounds 1–16, 1,538 push laps, 23 drivers, 11 cars.
PPC = percentile of the observed statistic among posterior-predictive replications (want 5–95%).

| Version | Change | Recovery | Divergences / max R-hat | LOO elpd (Δ vs previous) | PPC failures |
|---|---|---|---|---|---|
| v1 | ex-Gaussian, segment steps | ✅ 86–91% coverage | 0 / 1.008 | −1308.8 | teammate gap (4%) |
| v1-B | neutral car-share prior | ✅ | 0 / 1.008 | −1309.6 (−0.8 ± 1.5) | same as v1 |
| v1.1 | + within-segment evolution γ | ✅ 89–94% | 0 / 1.005 | −1079.5 (**+229 ± 21**) | teammate gap, skew, kurtosis |
| v1.2 | + clean / compromised mixture | ✅ 88–94% | 0 / 1.006 | −1037.2 (**+42 ± 13**) | teammate gap, field spread, Q1→Q2 (best-lap stats) |
| v1.3 | development vs track fit as total + share | ✅ 83–94% | 0 / 1.003 | −1.3 ± 1.0 vs v1.2 (level) | as v1.2 |

## Decisions
1. **Car share is data-driven.** v1 (informative prior, prior median 0.98) and v1-B (neutral prior, prior median 0.82) both give 0.98.
2. **v1.1 kept** despite the teammate-gap PPC still failing: residuals of v1 trend by −0.58% per 10 min within a segment (9 SE), and LOO improves by 229 ± 21. Point estimates unchanged (driver effects r = 0.986 with v1).
3. **v1.2 was production until v1.3.** Fixes skew and kurtosis (v1.1: 99th percentile), reduces Pareto k > 0.7 from 18 to 4 laps, LOO +42 ± 13. Point estimates unchanged (r = 0.992).

4. **v1.3 is production.** Same model, reparameterised: σ_car_event ~ N⁺(0, 0.6) and ρ_dev ~ Beta(2, 4), deriving σ_dev and σ_ν (marginal priors match v1.2). Round 14: R-hat 1.016 (warn) → 1.001 (pass); ESS of σ_dev 508 → 1,658 (round 14) and 936 → 1,603 (full season). All other estimates identical; LOO level.

## Known limitation
Best-lap statistics fail in every version: real best laps are tighter than "best of N independent laps",
because drivers adapt (abort, go again, stop after a good lap). Consequence: head-to-head and pole
probabilities are **conservative** (~20% too uncertain). Fix if needed: model each driver's best lap per
segment directly (v1.3).

## Open
- Best-lap calibration (v1.4).
- Simulation-based calibration across many simulated seasons (issue #3).
- Multi-season driver ratings, 2023–2026 (issue #7).
