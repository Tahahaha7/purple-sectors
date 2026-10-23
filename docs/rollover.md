# Season rollover runbook

Run once, between the last race of one season and the first qualifying session of the next.

```bash
git switch main && git pull
PYTHONPATH=src python -m purple_sectors.rollover --from 2026 --to 2027
git add -A && git commit -m "Roll over to the 2027 season" && git push
```

What happens:
- `site/archive/2026/` gets the final `soft.json`, `pace.json`, the full `snapshots/` history and the season's `forecasts/`.
- The Soft page keeps showing 2026 until the first 2027 fit replaces it; Medium and Inter show "refreshing" until then.
- `update.yml` and `backfill.yml` switch to `YEAR: "2027"`. The model keeps fitting every season from `FIRST_YEAR` (2023), so 2027 starts with four seasons of context and every driver's 2026 skill.

Before the first 2027 qualifying session, review: new teams' colours (`TEAM` in the site pages), new circuits' short codes (`CODE`), the team short names in `pace/export.py`, and run the History workflow once so `driver_debuts.json` knows the new rookies.
