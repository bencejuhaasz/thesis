import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

calls = api_fail = parse_fail = extracted = retried = 0
for path in sorted(RESULTS.glob("final_*/results.jsonl")):
    last = {}
    for line in path.open():
        if line.strip():
            j = json.loads(line)
            last[(j["sample_id"], j.get("repeat", 0))] = j
    for j in last.values():
        calls += 1
        api_fail += not j["api_ok"]
        parse_fail += not j.get("parse_ok")
        extracted += bool((j.get("pred") or {}).get("json_fallback"))
        if (j.get("retries") or 0) > 0:
            retried += 1
            print(f"retried: {path.parent.name} {j['sample_id']}, retries {j['retries']}, api_ok {j['api_ok']}")
print(f"final runs: calls {calls}, API error {api_fail}, unparsable {parse_fail}, "
      f"JSON extracted from text {extracted}, retried {retried}")

p = pd.read_csv(HERE / "plausibility" / "plausibility.csv")
d = pd.read_csv(HERE / "plausibility" / "flagged_items.csv")
print(f"\nplausibility check: meals {p.meals.sum()}, items {p['items'].sum()}, flagged items {len(d)}")
ratio = d.kcal_from_macros / d.kcal
gross = ratio > 5
excess = d.carb_g + d.protein_g + d.fat_g - d.weight_g
mass_only = d.mass_rule & ~d.energy_rule
below = ~gross & d.energy_rule & (ratio < 1)
above = ~gross & d.energy_rule & (ratio > 1)
print(f"only the mass rule: {mass_only.sum()} items, macronutrients exceed the item mass by "
      f"{excess[mass_only].min():.1f}-{excess[mass_only].max():.1f} g; {sorted(set(d[mass_only].item))}")
print(f"E_m < E: {below.sum()} items")
print(d[below][["run", "sample_id", "item", "weight_g", "kcal", "kcal_from_macros"]].to_string())
print(f"E_m / E {ratio[above].min():.2f}-{ratio[above].max():.2f}: {above.sum()} items; {sorted(set(d[above].item))}")
print(f"E_m / E > 5 ({ratio[gross].min():.1f}-{ratio[gross].max():.1f}): {gross.sum()} items")
print(d[gross].assign(ratio=ratio[gross].round(1)).drop(columns=["mass_rule", "energy_rule"]).to_string())
print(f"largest E_m / E outside the > 5 group: {ratio[~gross].max():.2f}")
