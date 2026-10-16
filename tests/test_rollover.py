import json
from pathlib import Path

from purple_sectors.rollover import rollover


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "site/data/snapshots").mkdir(parents=True)
    (tmp_path / "site/data/soft.json").write_text(json.dumps({"round": 24}))
    (tmp_path / "site/data/snapshots/r24.json").write_text("{}")
    wf = tmp_path / ".github/workflows"; wf.mkdir(parents=True)
    for f in ["update.yml", "backfill.yml"]:
        (wf / f).write_text('env:\n  YEAR: "2026"\n  CMDSTAN_VERSION: "2.36.0"\n')
    return tmp_path


def test_rollover_archives_and_switches_year(tmp_path):
    root = _repo(tmp_path)
    arch = rollover(2026, 2027, root)
    assert (arch / "soft.json").exists() and (root / "site/data/soft.json").exists()   # copied: site keeps last season
    assert (arch / "snapshots/r24.json").exists() and not (root / "site/data/snapshots").exists()
    assert 'YEAR: "2027"' in (root / ".github/workflows/update.yml").read_text()


def test_rollover_refuses_to_run_twice(tmp_path):
    root = _repo(tmp_path)
    rollover(2026, 2027, root)
    try:
        rollover(2026, 2027, root)
    except SystemExit:
        return
    raise AssertionError("second rollover should have been refused")
