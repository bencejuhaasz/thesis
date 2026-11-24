from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "final_descriptive"
DATASETS = [("n5k", "Nutrition5k"), ("omni", "OmniFood8k"), ("acetada", "ACETADA"), ("dfood", "DFood-Tj")]
MODELS = ["lite", "sonnet", "sol"]
OTHER = ["kcal", "mass_g", "protein_g", "fat_g"]

a = pd.read_csv(OUT / "accuracy.csv").set_index(["run", "target"])
p = pd.read_csv(OUT / "pairs.csv")
p["detectable"] = (p.CI_lo > 0) | (p.CI_hi < 0)

print("pMAE +- half width of the 95% CI; carb skill")
print(f"{'':12s} {'':7s} " + " ".join(f"{k:16s}" for k in OTHER) + "carb skill")
for key, name in DATASETS:
    for m in MODELS:
        run = f"final_{key}_{m}_macros"
        cells = [f"{a.loc[(run, k)].pMAE:.1f} +- {(a.loc[(run, k)].pMAE_hi - a.loc[(run, k)].pMAE_lo) / 2:.1f}"
                 for k in OTHER]
        print(f"{name:12s} {m:7s} " + " ".join(f"{c:16s}" for c in cells) + f"{a.loc[(run, 'carb_g')].skill:.2f}")

print("\ncarb bias % (all meals / lower third of the reference)")
for key, name in DATASETS:
    print(f"{name:12s} " + "  ".join(
        f"{m} {a.loc[(f'final_{key}_{m}_macros', 'carb_g')]['bias%']:+.0f} / "
        f"{a.loc[(f'final_{key}_{m}_macros', 'carb_g')]['bias%_low3']:+.0f}" for m in MODELS))

lite = p[p.A.str.contains("_lite_") & p.target.isin(OTHER)]
print(f"\nFlash-Lite vs Sonnet / Sol, {len(lite)} differences: detectable {lite.detectable.sum()}, "
      f"Flash-Lite more accurate {(lite.CI_lo > 0).sum()}, the other model more accurate {(lite.CI_hi < 0).sum()}")
print(lite[lite.detectable].round(1).to_string(index=False))
acetada_sonnet = lite[(lite.dataset == "acetada") & lite.B.str.contains("_sonnet_")]
print(f"ACETADA, Flash-Lite vs Sonnet: detectable {acetada_sonnet.detectable.sum()} of {len(acetada_sonnet)}")

ss = p[p.A.str.contains("_sonnet_")]
print(f"\nSonnet vs Sol, {len(ss)} differences: detectable {ss.detectable.sum()}")
print(ss[ss.detectable].round(1).to_string(index=False))
print(ss[ss.target == "carb_g"].round(1).to_string(index=False))
