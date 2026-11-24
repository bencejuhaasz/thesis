import json
from pathlib import Path

import significance as sig

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATASETS = [("nutrition5k", "nutrition5k_macros_n50_s42", "nutrition5k_lite_{}_n50_s42"),
            ("omnifood8k", "omnifood8k_allimage_n50_s42", "omnifood8k_lite_{}_n50_s42"),
            ("snapme", "snapme_lite_macros_n50_s42", "snapme_lite_{}_n50_s42"),
            ("acetada", "acetada_lite_macros_n50_s42", "acetada_lite_{}_n50_s42"),
            ("dfood_tj", "dfood_lite_macros_oblique_n50_s42", "dfood_lite_{}_oblique_n50_s42")]
VARIANTS = [("P3_recipe", "recipe"), ("P3_anchor", "anchor"), ("P3_portion", "portion"),
            ("P3_check", "check"), ("P3_recipe_check", "recipecheck")]
TARGETS = ["carb_g", "fat_g", "kcal"]
CONFIRM = ("omnifood8k_lite_macros_confirm_s42", "omnifood8k_lite_recipe_confirm_s42", "fat_g")


def load(run):
    res = RESULTS / run / "results.jsonl"
    r = json.loads(res.open().readline())
    return sig.load_group(res, r["dataset"], r["model_label"], r["prompt_variant"], r["output_mode"])


def pmae(d, k):
    _, x, t = sig.metric_arrays(d, k, "pMAE")
    return sig.ratio_stats(x, t)[0]


def paired(dA, dB, k):
    r = sig.paired(dA, dB, k, "pMAE", 10)
    h = float(sig._student_t.ppf(0.975, r["n_common"] - 1)) * r["SE_diff_paired"]
    lo, hi = r["delta_pp"] - h, r["delta_pp"] + h
    return r, lo > 0 or hi < 0, f"{r['delta_pp']:.1f} ({lo:.1f}; {hi:.1f})"


detected = []
for ds, base_run, pattern in DATASETS:
    base = load(base_run)
    print(f"\n== {ds}: {len(base)} meals")
    print(f"{'P3_macros (pMAE)':18s} " + " ".join(f"{pmae(base, k):<22.1f}" for k in TARGETS))
    for name, tag in VARIANTS:
        d = load(pattern.format(tag))
        cells = []
        for k in TARGETS:
            _, det, cell = paired(base, d, k)
            cells.append(cell)
            if det:
                detected.append((ds, name, k, cell))
        print(f"{name:18s} " + " ".join(f"{c:22s}" for c in cells))
print(f"\n{'columns':18s} " + " ".join(f"{k:22s}" for k in TARGETS))
print("\ndetectable differences (95% CI without 0):")
for row in detected:
    print("  " + "  ".join(row))

dA, dB = load(CONFIRM[0]), load(CONFIRM[1])
done = set()
for run in [base_run for ds, base_run, _ in DATASETS if ds == "omnifood8k"] + \
        [p.format(t) for ds, _, p in DATASETS if ds == "omnifood8k" for _, t in VARIANTS]:
    done |= set(load(run)["unit"])
dA, dB = dA[~dA["unit"].isin(done)], dB[~dB["unit"].isin(done)]
r, _, cell = paired(dA, dB, CONFIRM[2])
print(f"\nconfirmatory test, OmniFood8k fat, P3_recipe vs P3_macros on {r['n_common']} new meals: "
      f"{r['A']:.1f} -> {r['B']:.1f}, difference {cell}, p = {r['p_paired']:.3f}")
