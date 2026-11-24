import contextlib
import io
import sys
import tempfile
from pathlib import Path

import pandas as pd

import ci
import significance as sig

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODELS = ["Gemini 3.5 Flash-Lite", "Gemini 3.8 Flash", "GPT-6 Luna", "GPT-6.1 Sol", "GPT-6 Astra",
          "Claude Haiku 5.5", "Claude Sonnet 5.5", "Claude Opus 5.5"]
RUNS = {
    "nutrition5k": ["nutrition5k_macros_n50_s42", "nutrition5k_flash38_macros_n50_s42",
                    "nutrition5k_gpt-luna_macros_n50_s42", "nutrition5k_sol_macros_n50_s42", None,
                    "nutrition5k_claude-haiku-low_macros_n50_s42", "nutrition5k_sonnet_macros_n50_s42",
                    "nutrition5k_claude-opus-low_macros_n50_s42"],
    "omnifood8k": ["omnifood8k_allimage_n50_s42", "omnifood8k_1image_flash38_n50_s42",
                   "omnifood8k_1image_gpt-luna_macros_n50_s42", "omnifood8k_1image_sol_n50_s42",
                   "omnifood8k_1image_astra_n50_s42", "omnifood8k_1image_claude-haiku-low_macros_n50_s42",
                   "omnifood8k_1image_sonnet_n50_s42", "omnifood8k_1image_claude-opus-low_macros_n50_s42"],
    "snapme": ["snapme_lite_macros_n50_s42", "snapme_flash38_macros_n50_s42", "snapme_gpt-luna_macros_n50_s42",
               "snapme_sol_macros_n50_s42", None, "snapme_claude-haiku-low_macros_n50_s42",
               "snapme_sonnet_macros_n50_s42", "snapme_claude-opus-low_macros_n50_s42"],
    "acetada": ["acetada_lite_macros_n50_s42", "acetada_flash38_macros_n50_s42", "acetada_gpt-luna_macros_n50_s42",
                "acetada_gpt-sol_macros_n50_s42", None, "acetada_claude-haiku-low_macros_n50_s42",
                "acetada_sonnet_macros_n50_s42", "acetada_claude-opus-low_macros_n50_s42"],
    "dfood_tj": ["dfood_lite_macros_oblique_n50_s42", "dfood_flash38_macros_oblique_n50_s42",
                 "dfood_gpt-luna_macros_oblique_n50_s42", "dfood_gpt-sol_macros_oblique_n50_s42", None,
                 "dfood_claude-haiku-low_macros_oblique_n50_s42", "dfood_sonnet_macros_oblique_n50_s42",
                 "dfood_claude-opus-low_macros_oblique_n50_s42"],
}


def ci_tables(runs):
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = ["ci.py", "--results", str(RESULTS), "--runs", *runs, "--out", tmp]
        with contextlib.redirect_stdout(io.StringIO()):
            ci.main()
        return (pd.read_csv(Path(tmp) / "summary_ci.csv").set_index("run"),
                pd.read_csv(Path(tmp) / "ci_long.csv"))


def load(summ, run):
    s = summ.loc[run]
    return sig.load_group(RESULTS / run / "results.jsonl", s.dataset, s.model, s.prompt, s.output)


def paired(dA, dB, k):
    r = sig.paired(dA, dB, k, "pMAE", 10)
    h = ci.t_crit(r["n_common"], 0.95) * r["SE_diff_paired"]
    return f"{r['delta_pp']:.1f} ({r['delta_pp'] - h:.1f}; {r['delta_pp'] + h:.1f})"


def interval(v, lo, hi, n):
    return f"{v:.{n}f} ({lo:.{n}f}; {hi:.{n}f})"


summ, long = ci_tables([r for rs in RUNS.values() for r in rs if r])
for ds, rs in RUNS.items():
    base = load(summ, rs[0])
    t = base["t_carb_g"].dropna().to_numpy(float)
    guess = long[(long.run == rs[0]) & (long.metric == "carb_g_skill")].guess_pMAE.iloc[0]
    print(f"\n== {ds}: meals {len(t)}, carb reference SD/mean {t.std() / t.mean():.2f}, "
          f"leave-one-out median guess carb pMAE {guess:.1f}")
    print(f"{'model':22s} {'energy pMAE':20s} {'carb pMAE':20s} {'carb skill':22s} {'USD/1000':>8s}  "
          f"{'carb: model - Lite':22s} {'energy: model - Lite':22s}")
    for name, run in zip(MODELS, rs):
        if run is None:
            continue
        s = summ.loc[run]
        d = load(summ, run)
        dc, de = ("", "") if run == rs[0] else (paired(base, d, "carb_g"), paired(base, d, "kcal"))
        print(f"{name:22s} {interval(s['kcal_pMAE%'], s.kcal_pMAE_CI_low, s.kcal_pMAE_CI_high, 1):20s} "
              f"{interval(s['carb_g_pMAE%'], s.carb_g_pMAE_CI_low, s.carb_g_pMAE_CI_high, 1):20s} "
              f"{interval(s.carb_g_skill, s.carb_g_skill_CI_low, s.carb_g_skill_CI_high, 2):22s} "
              f"{1000 * s.cost_usd / s.calls:8.2f}  {dc:22s} {de:22s}")
