# Season rollover runbook

Run once, between the last race of one season and the first qualifying session of the next.

```bash
git switch main && git pull
gh issue develop <issue> --checkout
PYTHONPATH=src python -m purple_sectors.rollover --from 2026 --to 2027
git add -A && git commit -m "Roll over to the 2027 season" && git push
gh pr create --fill && gh pr merge --squash --delete-branch
```

What happens:
- `site/archive/2026/` gets the final `soft.json`, `car_pace.json` and the full `snapshots/` history.
- The Soft page keeps showing 2026 until the first 2027 fit replaces it; Medium and Inter show "refreshing" until then.
- `update.yml` and `backfill.yml` switch to `YEAR: "2027"`. The OpenF1 cache is keyed by year, so nothing stale is reused.

Before the first 2027 qualifying session, review: new teams' colours (`TEAM` in the site pages), new circuits' short codes (`CODE`), and the priors if the regulations change.
