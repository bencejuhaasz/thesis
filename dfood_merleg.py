import json
import re
from pathlib import Path

import prereg_tests as P
import significance as sig

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = {"Flash-Lite": "final_dfood_lite_macros", "Sonnet": "final_dfood_sonnet_macros",
        "Sol": "final_dfood_sol_macros"}
SCALE = re.compile(r"scale[^.]{0,60}?\d+(\.\d+)?\s*g|\d+(\.\d+)?\s*g[^.]{0,40}scale|scale (reading|showing|display)",
                   re.I)
TARGETS = ["mass_g", "carb_g", "kcal"]

mentioned = {}
for name, run in RUNS.items():
    for line in (RESULTS / run / "results.jsonl").open():
        if line.strip():
            j = json.loads(line)
            if j.get("parse_ok") and SCALE.search((j.get("pred") or {}).get("description") or ""):
                mentioned.setdefault(str(j["sample_id"]), set()).add(name)

frames = {name: P.load_run(run, RESULTS)[0] for name, run in RUNS.items()}
tested = frames["Flash-Lite"].index.astype(str)
scale = set(tested) & set(mentioned)
print(f"tested meals {len(tested)}, a scale reading in the description of at least one model {len(scale)}")
for name in RUNS:
    print(f"  {name}: {sum(name in mentioned.get(s, ()) for s in tested)}")


def subsets(d):
    flag = d.index.astype(str).isin(list(scale))
    return {"all": d, "scale": d[flag], "no scale": d[~flag]}


print(f"\n{'model':11s} {'meals':9s} {'n':>4s} " + " ".join(f"{k + ' pMAE':>13s}" for k in TARGETS))
for name, d in frames.items():
    for part, sub in subsets(d).items():
        cells = []
        for k in TARGETS:
            _, x, t = sig.metric_arrays(sub, k, "pMAE")
            cells.append(f"{sig.ratio_stats(x, t)[0]:13.1f}")
        print(f"{name:11s} {part:9s} {len(sub):4d} " + " ".join(cells))

print("\nA family, DFood-Tj, carbohydrate: model - Flash-Lite, percentage points (no correction, descriptive)")
base = subsets(frames["Flash-Lite"])
for name in ("Sonnet", "Sol"):
    other = subsets(frames[name])
    for part in base:
        r = P.test(base[part], other[part], "carb_g")
        print(f"  {name:7s} {part:9s} n={r['n_pairs']:3d}  {r['pMAE_A']:5.1f} -> {r['pMAE_B']:5.1f}  "
              f"{r['delta_pp']:+5.1f} ({r['CI_lo']:+.1f}; {r['CI_hi']:+.1f})  p = {r['p']:.4f}")
