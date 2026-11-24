import json
from pathlib import Path

import pandas as pd

import significance as sig

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TARGETS = ["kcal", "carb_g", "fat_g"]
ROWS = [
    ("Több nézet (Flash-Lite)", [
        ("OmniFood8k: 6 kép, nem 1 (P3_macros)", "omnifood8k_allimage_n50_s42", "omnifood8k_1image_n50_s42"),
        ("DFood-Tj: ferde és felülnézet, nem csak ferde (P3_macros)", "dfood_lite_macros_oblique_n50_s42",
         "macros50_35"),
        ("DFood-Tj: felülnézet, nem ferde (P3)", "new_oblique_dfood_tj_n50_s42_gemini_flash_lite_pre",
         "new_overhead_dfood_tj_n50_s42_gemini_flash_lite_pre"),
        ("DFood-Tj: ferde és felülnézet, nem csak ferde (P3)", "new_oblique_dfood_tj_n50_s42_gemini_flash_lite_pre",
         "new_both_dfood_tj_n50_s42_gemini_flash_lite_pre"),
    ]),
    ("P3_prep a P3_macros-hoz képest (Flash-Lite, 1 kép)", [
        ("Nutrition5k", "nutrition5k_macros_n50_s42", "nutrition5k_lite_prep_n50_s42"),
        ("OmniFood8k", "omnifood8k_allimage_n50_s42", "omnifood8k_lite_prep_n50_s42"),
        ("SNAPMe", "snapme_lite_macros_n50_s42", "snapme_lite_prep_n50_s42"),
    ]),
    ("P3_prep_textcalc a P3_prep-hez képest (Flash-Lite, 1 kép)", [
        ("Nutrition5k", "nutrition5k_lite_prep_n50_s42", "nutrition5k_lite_preptextcalc_n50_s42"),
        ("OmniFood8k", "omnifood8k_lite_prep_n50_s42", "omnifood8k_lite_preptextcalc_n50_s42"),
        ("SNAPMe", "snapme_lite_prep_n50_s42", "snapme_lite_preptextcalc_n50_s42"),
    ]),
    ("P3_density", [
        ("OmniFood8k, 6 kép, a P3_macros-hoz képest (Flash-Lite)", "omnifood8k_1image_n50_s42",
         "omnifood8k_allimage_density_n50_s42"),
        ("DFood-Tj, 2 nézet, a P3-hoz képest (Gemini 3.1 Flash-Lite)",
         "new_both_dfood_tj_n350_s42_gemini_3.1_flash_lite_pre", "dens50"),
    ]),
]


def load(run):
    res = RESULTS / run / "results.jsonl"
    r = json.loads(res.open().readline())
    return sig.load_group(res, r["dataset"], r["model_label"], r["prompt_variant"], r["output_mode"])


def summary(run):
    return pd.read_csv(RESULTS / run / "summary.csv").iloc[0]


def paired(dA, dB, k):
    if f"p_{k}" not in dA or f"p_{k}" not in dB:
        return "--"
    r = sig.paired(dA, dB, k, "pMAE", 10)
    h = float(sig._student_t.ppf(0.975, r["n_common"] - 1)) * r["SE_diff_paired"]
    return f"{r['delta_pp']:.1f} ({r['delta_pp'] - h:.1f}; {r['delta_pp'] + h:.1f}) n={r['n_common']}"


def latency_ratio(dA, dB):
    a, b = dA.set_index("unit"), dB.set_index("unit")
    common = a.index.intersection(b.index)
    return b.loc[common, "latency_s"].median() / a.loc[common, "latency_s"].median()


def cost_ratio(sA, sB):
    if not (sA.cost_usd > 0 and sB.cost_usd > 0):
        return "--"
    return f"{(sB.cost_usd / sB.calls) / (sA.cost_usd / sA.calls):.2f}"


print(f"{'':60s} " + " ".join(f"{k:26s}" for k in TARGETS) + "cost (x)  latency (x)")
for group, rows in ROWS:
    print(f"\n{group}")
    for name, a, b in rows:
        dA, dB = load(a), load(b)
        sA, sB = summary(a), summary(b)
        print(f"  {name:58s} " + " ".join(f"{paired(dA, dB, k):26s}" for k in TARGETS)
              + f"{cost_ratio(sA, sB):9s} {latency_ratio(dA, dB):.2f}")
