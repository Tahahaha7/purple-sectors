"""Combine the outputs of the "Compare models" workflow into one table (job summary + JSON).

Usage:  python -m purple_sectors.compare --dir artifacts --base quali_v12 --candidate quali_v13
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from purple_sectors.loo import compare


def main(d: Path, base: str, cand: str) -> dict:
    reps = {p.stem: json.loads(p.read_text()) for p in d.rglob("report_*.json")}
    rows = []
    for key, r in sorted(reps.items()):
        model = Path(r["stan"]).stem
        rows.append({"model": model, "rounds": r["max_round"] or "all", **{k: r[k] for k in ["divergences", "draws", "max_rhat", "worst", "status", "car_share"]},
                     "ess_sigma_dev": r["ess"].get("sigma_dev"), "med_sigma_dev": r["medians"].get("sigma_dev"),
                     "med_sigma_nu": r["medians"].get("sigma_nu")})
    out = {"fits": rows}
    a, b = next(d.rglob(f"post_{base}_all.npz"), None), next(d.rglob(f"post_{cand}_all.npz"), None)
    if a and b:
        out["loo"] = compare(np.load(a)["log_lik"], np.load(b)["log_lik"])
    rec = next(d.rglob("recovery.json"), None)
    if rec:
        out["recovery"] = json.loads(rec.read_text())
    print(json.dumps(out, indent=1))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        L = [f"## {cand} vs {base}", "| model | rounds | status | divergences | max R-hat (worst) | ESS σ_dev | σ_dev | σ_ν | car share |",
             "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (str(r["rounds"]), r["model"])):
            L.append(f"| {r['model']} | {r['rounds']} | {r['status']} | {r['divergences']}/{r['draws']} | {r['max_rhat']:.3f} (`{r['worst']}`) | "
                     f"{r['ess_sigma_dev']} | {r['med_sigma_dev']:.3f} | {r['med_sigma_nu']:.3f} | {r['car_share']:.3f} |")
        if "loo" in out:
            l = out["loo"]
            L += ["", f"**LOO (all rounds):** {cand} − {base} = {l['diff']:+.1f} ± {l['se']:.1f} "
                      f"(Pareto k > 0.7: {l['k_bad_a']} → {l['k_bad_b']} laps)"]
        Path(summary).open("a").write("\n".join(L) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--base", default="quali_v12")
    ap.add_argument("--candidate", default="quali_v13")
    a, _ = ap.parse_known_args()
    main(a.dir, a.base, a.candidate)
