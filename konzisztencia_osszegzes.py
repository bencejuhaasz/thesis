from pathlib import Path

import numpy as np
import pandas as pd

import consistency_pairs as CP

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUTLIER = ("final_omni_sonnet_macros", "fat_g")

d = pd.read_csv(HERE / "consistency_pairs" / "consistency_pairs.csv")
d["model"] = d.final.str.extract(r"_(lite|sonnet|sol)_")[0]
d["prompt"] = d.final.str.rsplit("_", n=1).str[1]
table = d[d.target.isin(["kcal", "carb_g"])].pivot_table(index=["dataset", "model", "prompt", "meals"],
                                                          columns="target", values="within_%")
print(table[["kcal", "carb_g"]].to_string())

for k in ("kcal", "carb_g"):
    s = d[d.target == k]
    print(f"\n{k}: within % {s['within_%'].min()}-{s['within_%'].max()}, largest: "
          f"{s.loc[s['within_%'].idxmax(), 'final']}")
carb = d[d.target == "carb_g"]
print("\ncarb within % by dataset:")
print(carb.groupby("dataset")["within_%"].agg(["min", "max"]).to_string())
print(f"carb within SD (g): {carb.within_SD.min()}-{carb.within_SD.max()}")

print("\ncarb reference mean (g) of the shared meals:")
for final, earlier, _ in pd.read_csv(HERE / "consistency_pairs" / "pairs_used.csv").itertuples(index=False):
    units, ds, view = CP.excluded_units(final, RESULTS)
    fp, ep = CP.preds(RESULTS / final, ds, view), CP.preds(RESULTS / earlier, ds, view)
    shared = [u for u in sorted(units) if u in fp and u in ep]
    print(f"  {final:28s} {len(shared):3d} {np.mean([fp[u][1]['carb_g'] for u in shared]):6.1f}")

print("\nwithin %, Flash-Lite vs Sonnet (P3_macros):")
lite_sonnet = d[d.target.isin(["kcal", "carb_g"]) & (d.prompt == "macros") & d.model.isin(["lite", "sonnet"])]
print(lite_sonnet.pivot_table(index=["dataset", "target"], columns="model", values="within_%").to_string())

rest = d[~((d.final == OUTLIER[0]) & (d.target == OUTLIER[1]))].assign(gap=lambda x: (x.pMAE_r0 - x.pMAE_r1).abs())
print(f"\nlargest |pMAE difference| of the two call series (without {OUTLIER[0]} {OUTLIER[1]}):")
print(rest.sort_values("gap").tail(3)[["final", "target", "pMAE_r0", "pMAE_r1", "gap"]].to_string())
ratio = carb["within_%"] / carb.pMAE_r0
print(f"\ncarb within % / pMAE of the final run on the same meals: {100 * ratio.min():.0f}-{100 * ratio.max():.0f}%")
print("\n" + d[(d.final == OUTLIER[0]) & (d.target == OUTLIER[1])][
    ["final", "earlier", "meals", "within_%", "pMAE_r0", "pMAE_r1"]].to_string())
