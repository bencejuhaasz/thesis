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
NUTRIBENCH = [("Gemini 3.5 Flash-Lite", "nutribench_natural_lite_n50_s42"),
              ("Gemini 3.8 Flash", "nutribench_natural_38low_n50_s42"),
              ("GPT-6 Luna", "nutribench_natural_gpt-luna_n50_s42"),
              ("GPT-6.1 Sol", "nutribench_natural_sol_n50_s42"),
              ("Claude Haiku 5.5", "nutribench_natural_claude-haiku-low_n50_s42"),
              ("Claude Sonnet 5.5", "nutribench_natural_sonnet_n50_s42"),
              ("Claude Opus 5.5", "nutribench_natural_claude-opus-low_n50_s42")]
HINT = [("Flash-Lite", "snapme_lite_macros_n50_s42", "snapme_lite_hint_n50_s42"),
        ("Sonnet", "snapme_sonnet_macros_n50_s42", "snapme_sonnet_hint_n50_s42"),
        ("Sol", "snapme_sol_macros_n50_s42", "snapme_sol_hint_n50_s42")]


def ci_summary(runs):
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = ["ci.py", "--results", str(RESULTS), "--runs", *runs, "--out", tmp]
        with contextlib.redirect_stdout(io.StringIO()):
            ci.main()
        return pd.read_csv(Path(tmp) / "summary_ci.csv").set_index("run")


def load(summ, run):
    s = summ.loc[run]
    return sig.load_group(RESULTS / run / "results.jsonl", s.dataset, s.model, s.prompt, s.output)


def paired(dA, dB, k):
    r = sig.paired(dA, dB, k, "pMAE", 10)
    h = ci.t_crit(r["n_common"], 0.95) * r["SE_diff_paired"]
    lo, hi = r["delta_pp"] - h, r["delta_pp"] + h
    return f"{r['delta_pp']:.1f} ({lo:.1f}; {hi:.1f}){' *' if lo > 0 or hi < 0 else ''}"


def interval(v, lo, hi, n):
    return f"{v:.{n}f} ({lo:.{n}f}; {hi:.{n}f})"


summ = ci_summary([r for _, r in NUTRIBENCH])
base = load(summ, NUTRIBENCH[0][1])
print("NutriBench (* = 95% CI without 0)")
print(f"{'model':22s} {'energy pMAE':20s} {'carb pMAE':20s} {'carb skill':20s} {'USD/1000':>8s}  "
      f"{'energy: model - Lite':24s} {'carb: model - Lite':24s}")
for name, run in NUTRIBENCH:
    s = summ.loc[run]
    d = load(summ, run)
    diffs = ("", "") if run == NUTRIBENCH[0][1] else (paired(base, d, "kcal"), paired(base, d, "carb_g"))
    print(f"{name:22s} {interval(s['kcal_pMAE%'], s.kcal_pMAE_CI_low, s.kcal_pMAE_CI_high, 1):20s} "
          f"{interval(s['carb_g_pMAE%'], s.carb_g_pMAE_CI_low, s.carb_g_pMAE_CI_high, 1):20s} "
          f"{interval(s.carb_g_skill, s.carb_g_skill_CI_low, s.carb_g_skill_CI_high, 2):20s} "
          f"{1000 * s.cost_usd / s.calls:8.2f}  {diffs[0]:24s} {diffs[1]:24s}")

summ = ci_summary([r for _, a, b in HINT for r in (a, b)])
print("\nSNAPMe, P3_macros_hint - P3_macros (* = 95% CI without 0)")
print(f"{'model':12s} {'fat':26s} {'carb':26s}")
for name, a, b in HINT:
    dA, dB = load(summ, a), load(summ, b)
    print(f"{name:12s} {paired(dA, dB, 'fat_g'):26s} {paired(dA, dB, 'carb_g'):26s}")
