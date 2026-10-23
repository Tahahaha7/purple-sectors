"""Season rollover: archive a finished season's site data and point the pipeline at the next year.

Usage:  python -m purple_sectors.rollover --from 2026 --to 2027

- copies soft.json and pace.json to site/archive/<old>/ (the site keeps showing them until
  the new season's first fit overwrites them)
- moves site/data/snapshots/ and site/data/forecasts/ to the archive (the history must not mix seasons)
- switches YEAR in the update and backfill workflows (the model keeps using every season since FIRST_YEAR)
Raw lap data in data/raw/ is kept as is: it is already one file per year.
"""
import argparse
import re
import shutil
from pathlib import Path

WORKFLOWS = [".github/workflows/update.yml", ".github/workflows/backfill.yml"]


def rollover(old: int, new: int, root: Path = Path(".")) -> Path:
    if new != old + 1:
        raise SystemExit(f"--to must be the year after --from ({old + 1}), got {new}")
    data, archive = root / "site" / "data", root / "site" / "archive" / str(old)
    if archive.exists():
        raise SystemExit(f"{archive} already exists: has {old} been rolled over already?")
    archive.mkdir(parents=True)
    for name in ["soft.json", "pace.json"]:
        if (data / name).exists():
            shutil.copy2(data / name, archive / name)
    for folder in ["snapshots", "forecasts"]:
        if (data / folder).exists():
            shutil.move(str(data / folder), str(archive / folder))
    for wf in WORKFLOWS:
        p = root / wf
        if p.exists():
            text, n = re.subn(rf'YEAR: "{old}"', f'YEAR: "{new}"', p.read_text())
            if n != 1:
                raise SystemExit(f'expected exactly one YEAR: "{old}" in {wf}, found {n}')
            p.write_text(text)
    print(f"Archived {old} to {archive}; workflows now use YEAR={new}.")
    print("Next: commit, push, then run the update workflow once the first qualifying session is done.")
    return archive


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="old", type=int, required=True)
    ap.add_argument("--to", dest="new", type=int, required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    a, _ = ap.parse_known_args()
    rollover(a.old, a.new, a.root)
