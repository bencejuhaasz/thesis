#!/usr/bin/env python3
"""
ci.py - 95% confidence intervals for every run in results/ (no API calls).

Put it next to benchmark.py AND significance.py (same loader and formulas).
    python ci.py                                 # every results/*/summary.csv
    python ci.py --exclude smoke dry             # skip runs whose name contains these
    python ci.py --runs runA runB                # only these runs
    python ci.py --bootstrap 5000                # + percentile bootstrap CI (normality check)
    python ci.py --level 0.90                    # other confidence level

Per run (summary.csv row) and target (kcal, mass_g, carb_g, protein_g, fat_g):
    pMAE%   = 100 * sum|p - t| / sum t     CI = pMAE +- t_crit * SE
    MAE     (units)                         CI = the pMAE CI x mean truth / 100
    bias%   = 100 * sum(p - t) / sum t      CI as above, + p value of "bias = 0"
            (CI not containing 0 <=> systematic under/over-estimation, p < 1 - level)
    SE: delta method on the per-meal errors (the deviation of estimate and control
    value), Student t with df = n - 1, i.e. normal assumption for the MEAN error.
    --bootstrap N adds the percentile bootstrap CI (resampling meals, no normality
    assumption). If both agree, the normal assumption is fine.
    latency_mean_s: mean +- t * SD / sqrt(n).
    skill   = 1 - sum|p - t| / sum|m_i - t|, m_i = median truth of the OTHER meals of the run
            (leave-one-out "always say the typical value" guess = the dataset's own spread).
            1 = perfect, 0 = no better than that guess, < 0 = worse. CI by the delta method
            on the ratio, p value of "skill = 0" (two-sided).
Also per run and target, how far ONE meal can be off (not a CI, the spread of the
individual errors): the 5th and 95th percentile of the per-meal % error 100*(p-t)/t
-> "90% of meals are between X% and Y%".

Output (--out, default ci/):
    ci_long.csv      one row per run x metric: value, SE, CI_low, CI_high (+ boot), p_bias0 / p_skill0
    skill_by_model.csv  skill (with CI and p) per model setup x dataset x target, one row each;
                     also printed as a model x dataset table per target
    summary_ci.csv   every summary.csv row, ORIGINAL columns unchanged, appended:
                     <t>_pMAE_CI_low/high, <t>_MAE_CI_low/high, <t>_bias%, <t>_bias%_CI_low/high,
                     <t>_bias_p, <t>_meal_err_p5/p95, <t>_skill, <t>_skill_CI_low/high, <t>_skill_p,
                     latency_mean_CI_low/high
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import significance as sig
except ImportError:
    raise SystemExit("significance.py must be in the same folder as ci.py")


def t_crit(n, level):
    try:
        from scipy.stats import t
        return float(t.ppf(0.5 + level / 2, max(n - 1, 1)))
    except Exception:
        from statistics import NormalDist
        return NormalDist().inv_cdf(0.5 + level / 2)


def boot_ci(x, t, level, B, rng):
    n = len(t)
    idx = rng.integers(0, n, size=(B, n))
    vals = 100 * x[idx].sum(1) / t[idx].sum(1)
    lo, hi = np.percentile(vals, [100 * (0.5 - level / 2), 100 * (0.5 + level / 2)])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--exclude", nargs="+", default=[])
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N")
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--out", default="ci")
    a = ap.parse_args()

    res_dir = Path(a.results)
    files = ([res_dir / r / "summary.csv" for r in a.runs] if a.runs
             else sorted(res_dir.glob("*/summary.csv")))
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"not found: {missing}")
    files = [f for f in files if not any(x in f.parent.name for x in a.exclude)]
    if not files:
        raise SystemExit(f"no summary.csv under {res_dir}")
    summ = sig.read_summaries([str(f) for f in files])
    rng = np.random.default_rng(0)
    pct = f"{100 * a.level:g}%"
    print(f"{pct} CI: delta method + Student t" + (f", bootstrap {a.bootstrap}x" if a.bootstrap else "")
          + f"; p: {sig.P_METHOD}\n")

    long_rows, extra_rows = [], []
    for i, row in summ.iterrows():
        extra = {}
        res = sig.results_path(row["run"], row["_source"], res_dir)
        d = None if res is None else sig.load_group(res, row["dataset"], row["model"], row["prompt"],
                                                    row.get("output", "?") or "?")
        if d is None or len(d) < a.min_n:
            print(f"  skip {row['run']} / {row['dataset']} / {row['model']}: "
                  + ("no results.jsonl / records" if d is None else f"only {len(d)} meals"))
            extra_rows.append(extra)
            continue
        head = {"run": row["run"], "dataset": row["dataset"], "model": row["model"],
                "prompt": row["prompt"], "images_per_call": row.get("images_per_call", "")}
        print(f"{row['run']} | {row['dataset']} | {row['model']} | {row['prompt']}")
        for k in sig.TARGETS:
            for kind in ("pMAE", "bias"):
                arr = sig.metric_arrays(d, k, kind)
                if arr is None:
                    continue
                _, x, t = arr
                n = len(t)
                v, se, _ = sig.ratio_stats(x, t)
                tc = t_crit(n, a.level)
                lo, hi = v - tc * se, v + tc * se
                rec = {**head, "metric": sig.mname(k, kind), "n": n, "value": v, "SE": se,
                       "CI_low": lo, "CI_high": hi}
                if a.bootstrap:
                    rec["boot_low"], rec["boot_high"] = boot_ci(x, t, a.level, a.bootstrap, rng)
                if kind == "bias":
                    z = v / se if se > 0 else float("nan")
                    rec["p_bias0"] = sig.p_two_sided(z, n) if z == z else float("nan")
                    pos = t > 0                    # meals with true value 0 have no % error
                    e_pct = 100 * x[pos] / t[pos]
                    p5, p95 = (np.percentile(e_pct, [5, 95]) if len(e_pct) else (np.nan, np.nan))
                    rec["meal_err_p5"], rec["meal_err_p95"] = float(p5), float(p95)
                    extra[f"{k}_bias%"] = round(v, 1)
                    extra[f"{k}_bias%_CI_low"], extra[f"{k}_bias%_CI_high"] = round(lo, 1), round(hi, 1)
                    extra[f"{k}_bias_p"] = float(f"{rec['p_bias0']:.3g}") if rec["p_bias0"] == rec["p_bias0"] else ""
                    extra[f"{k}_meal_err_p5"], extra[f"{k}_meal_err_p95"] = round(p5, 1), round(p95, 1)
                else:
                    tm = t.mean()
                    extra[f"{k}_pMAE_CI_low"], extra[f"{k}_pMAE_CI_high"] = round(lo, 1), round(hi, 1)
                    extra[f"{k}_MAE_CI_low"] = round(lo * tm / 100, 2)
                    extra[f"{k}_MAE_CI_high"] = round(hi * tm / 100, 2)
                long_rows.append(rec)
            arr = sig.metric_arrays(d, k, "pMAE")
            if arr is not None and len(arr[2]) >= 3:
                _, ae, t = arr
                n = len(t)
                med = np.array([np.median(np.delete(t, j)) for j in range(n)])
                bg = np.abs(med - t)
                if bg.sum() > 0:
                    R = ae.sum() / bg.sum()
                    u = ae - R * bg
                    se = math.sqrt((u ** 2).sum() / (n * (n - 1))) / bg.mean()
                    tc = t_crit(n, a.level)
                    sk, lo, hi = 1 - R, 1 - R - tc * se, 1 - R + tc * se
                    z = sk / se if se > 0 else float("nan")
                    pz = sig.p_two_sided(z, n) if z == z else float("nan")
                    long_rows.append({**head, "metric": f"{k}_skill", "n": n, "value": sk, "SE": se,
                                      "CI_low": lo, "CI_high": hi, "p_skill0": pz,
                                      "guess_pMAE": 100 * bg.sum() / t.sum()})
                    extra[f"{k}_skill"] = round(sk, 3)
                    extra[f"{k}_skill_CI_low"], extra[f"{k}_skill_CI_high"] = round(lo, 3), round(hi, 3)
                    extra[f"{k}_skill_p"] = float(f"{pz:.3g}") if pz == pz else ""
            m = next((r for r in reversed(long_rows) if r["run"] == row["run"]
                      and r["dataset"] == row["dataset"] and r["model"] == row["model"]
                      and r["prompt"] == row["prompt"] and r["metric"] == f"{k}_pMAE%"), None)
            b = next((r for r in reversed(long_rows) if r["run"] == row["run"]
                      and r["dataset"] == row["dataset"] and r["model"] == row["model"]
                      and r["prompt"] == row["prompt"] and r["metric"] == f"{k}_bias%"), None)
            if m and b:
                bt = f"  boot [{m['boot_low']:5.1f}; {m['boot_high']:5.1f}]" if a.bootstrap else ""
                s = next((r for r in reversed(long_rows) if r["run"] == row["run"]
                          and r["dataset"] == row["dataset"] and r["model"] == row["model"]
                          and r["prompt"] == row["prompt"] and r["metric"] == f"{k}_skill"), None)
                st = (f"   skill {s['value']:+.2f} [{s['CI_low']:+.2f}; {s['CI_high']:+.2f}] "
                      f"(guess {s['guess_pMAE']:.1f}%)" if s else "")
                print(f"  {k:10s} pMAE {m['value']:5.1f} [{m['CI_low']:5.1f}; {m['CI_high']:5.1f}]{bt}   "
                      f"bias {b['value']:+6.1f}% [{b['CI_low']:+6.1f}; {b['CI_high']:+6.1f}] "
                      f"p {b['p_bias0']:.2g}{sig.stars(b['p_bias0'])}   "
                      f"one meal (90%): {b['meal_err_p5']:+.0f}% .. {b['meal_err_p95']:+.0f}%{st}")
        if "latency_s" in d and d["latency_s"].notna().sum() > 2:
            L = d["latency_s"].dropna().to_numpy(float)
            tc = t_crit(len(L), a.level)
            se = L.std(ddof=1) / math.sqrt(len(L))
            long_rows.append({**head, "metric": "latency_mean_s", "n": len(L), "value": L.mean(), "SE": se,
                              "CI_low": L.mean() - tc * se, "CI_high": L.mean() + tc * se})
            extra["latency_mean_CI_low"] = round(L.mean() - tc * se, 2)
            extra["latency_mean_CI_high"] = round(L.mean() + tc * se, 2)
        print()
        extra_rows.append(extra)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(long_rows)
    for c in ("value", "SE", "CI_low", "CI_high", "boot_low", "boot_high", "meal_err_p5", "meal_err_p95"):
        if c in long:
            long[c] = long[c].round(2)
    for c in ("p_bias0", "p_skill0"):
        if c in long:
            long[c] = long[c].map(lambda v: float(f"{v:.3g}") if v == v else v)
    if "guess_pMAE" in long:
        long["guess_pMAE"] = long["guess_pMAE"].round(2)
    long.to_csv(out / "ci_long.csv", index=False)

    orig = [c for c in summ.columns if c != "_source"]
    ex = pd.DataFrame(extra_rows, index=summ.index)
    order = []
    for k in sig.TARGETS:
        order += [c for c in (f"{k}_pMAE_CI_low", f"{k}_pMAE_CI_high", f"{k}_MAE_CI_low", f"{k}_MAE_CI_high",
                              f"{k}_bias%", f"{k}_bias%_CI_low", f"{k}_bias%_CI_high", f"{k}_bias_p",
                              f"{k}_meal_err_p5", f"{k}_meal_err_p95", f"{k}_skill", f"{k}_skill_CI_low",
                              f"{k}_skill_CI_high", f"{k}_skill_p") if c in ex]
    order += [c for c in ("latency_mean_CI_low", "latency_mean_CI_high") if c in ex]
    pd.concat([summ[orig], ex[order]], axis=1).to_csv(out / "summary_ci.csv", index=False)
    # --- skill per model setup x dataset
    sk = long[long["metric"].str.endswith("_skill")].copy() if len(long) else pd.DataFrame()
    if len(sk):
        def ipc(v):
            try:
                return f"{float(v):g} img"
            except (TypeError, ValueError):
                return "? img"
        sk["setup"] = sk["model"] + " | " + sk["prompt"] + " | " + sk["images_per_call"].map(ipc)
        sk["target"] = sk["metric"].str[:-len("_skill")]
        sk[["run", "setup", "dataset", "target", "n", "value", "CI_low", "CI_high", "p_skill0", "guess_pMAE"]] \
            .rename(columns={"value": "skill"}).to_csv(out / "skill_by_model.csv", index=False)
        print(f"\nSKILL per model setup (1 = perfect, 0 = no better than guessing the typical value; "
              f"* = differs from 0, p < 0.05). Several runs of one setup on a dataset: the one with most meals.")
        sk = sk.sort_values("n").drop_duplicates(["setup", "dataset", "target"], keep="last")
        for tg in sig.TARGETS:
            g = sk[sk["target"] == tg]
            if g.empty:
                continue
            tab = g.assign(cell=[f"{v:+.2f}{'*' if p == p and p < 0.05 else ' '}"
                                 for v, p in zip(g["value"], g["p_skill0"])]) \
                   .pivot(index="setup", columns="dataset", values="cell").fillna("")
            print(f"\n[{tg}]")
            print(tab.to_string())
    print(f"\n-> {out / 'ci_long.csv'}, {out / 'summary_ci.csv'}, {out / 'skill_by_model.csv'}")


if __name__ == "__main__":
    main()
