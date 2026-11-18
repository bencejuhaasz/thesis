#!/usr/bin/env python3
"""
final_descriptive.py - the descriptive (not tested) results of the final round, as listed
in notes/prereg_final_2026-11-12.md ("Leíró eredmények"). No API calls.
Put it next to prereg_tests.py and significance.py; it uses the same loader and the same
tested positions (the earlier, already analysed positions are excluded).

    venv/bin/python final_descriptive.py

Output (folder final_descriptive/):
  accuracy.csv  per arm x target: n (parsed meals), n_calls, success_rate, pMAE_all (failed
                call counted as estimate 0) and pMAE (parsed calls only) and bias% with
                95% CI (delta method, Student t),
                slope (pred on truth), skill vs the leave-one-out median guess, and the
                bias% in the lower / middle / upper third of the true values
  pairs.csv     paired differences with 95% CI, descriptive only (no decision, no Holm):
                the A-block model pairs for kcal, mass_g, protein_g, fat_g, and
                Sonnet vs Sol for every target
  ops.csv       per arm: calls, api_fail, parse_fail (summary.csv, all positions),
                calls_tested, failed_tested (tested positions),
                latency mean / p50 on the tested positions, cost per 1000 calls
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import significance as sig  # noqa: E402
import prereg_tests as P  # noqa: E402
from scipy.stats import t as _t  # noqa: E402

TARGETS = ["kcal", "mass_g", "carb_g", "protein_g", "fat_g"]
ARMS = sorted({r for fam in P.FAMILIES.values() for A, B, _ in fam for r in (A, B)})
OTHER = ["kcal", "mass_g", "protein_g", "fat_g"]
PAIRS = ([(f"final_{d}_lite_macros", f"final_{d}_{o}_macros", k)
          for d in ("n5k", "omni", "acetada", "dfood") for o in ("sonnet", "sol") for k in OTHER]
         + [(f"final_{d}_sonnet_macros", f"final_{d}_sol_macros", k)
            for d in ("n5k", "omni", "acetada", "dfood") for k in TARGETS])


def ci(v, se, n):
    h = float(_t.ppf(0.975, n - 1)) * se
    return v - h, v + h


def accuracy(run, d):
    rows = []
    for k in TARGETS:
        a = sig.metric_arrays(d, k, "pMAE")
        if a is None:
            continue
        idx = a[0]
        p = d.loc[idx, f"p_{k}"].to_numpy(float)
        t = d.loc[idx, f"t_{k}"].to_numpy(float)
        n = len(t)
        v, se, _ = sig.ratio_stats(np.abs(p - t), t)
        b, seb, _ = sig.ratio_stats(p - t, t)
        loo_med = np.array([np.median(np.delete(t, i)) for i in range(n)])
        third = np.digitize(t, np.quantile(t, [1 / 3, 2 / 3]))
        terc = [100 * (p[third == j] - t[third == j]).sum() / t[third == j].sum() for j in range(3)]
        lo, hi = ci(v, se, n)
        blo, bhi = ci(b, seb, n)
        za = sig.metric_arrays_all(d, k)          # F2: failed call = estimate 0
        n_all = len(za[2])
        va, sea, _ = sig.ratio_stats(za[1], za[2])
        alo, ahi = ci(va, sea, n_all)
        rows.append({"run": run, "target": k, "n": n, "n_calls": n_all,
                     "success_rate": n / n_all, "pMAE_all": va, "pMAE_all_lo": alo,
                     "pMAE_all_hi": ahi, "pMAE": v, "pMAE_lo": lo, "pMAE_hi": hi,
                     "bias%": b, "bias_lo": blo, "bias_hi": bhi,
                     "slope": np.polyfit(t, p, 1)[0],
                     "skill": 1 - np.abs(p - t).sum() / np.abs(loo_med - t).sum(),
                     "bias%_low3": terc[0], "bias%_mid3": terc[1], "bias%_high3": terc[2]})
    return rows


def main():
    results = HERE / "results"
    cache = {r: P.load_run(r, results) for r in ARMS}

    acc = pd.DataFrame([row for r in ARMS for row in accuracy(r, cache[r][0])])
    acc.insert(1, "dataset", acc["run"].map(lambda r: cache[r][1]))

    pairs = []
    for A, B, k in PAIRS:
        res = P.test(cache[A][0], cache[B][0], k)
        pairs.append({"dataset": cache[A][1], "A": A, "B": B, "target": k,
                      **{c: res[c] for c in ("n_pairs", "pMAE_A", "pMAE_B", "delta_pp", "CI_lo", "CI_hi")}})
    pairs = pd.DataFrame(pairs)

    ops = []
    for r in ARMS:
        d = cache[r][0]
        s = pd.read_csv(results / r / "summary.csv").iloc[0]
        lat = d["latency_s"].dropna() if "latency_s" in d else pd.Series(dtype=float)
        ops.append({"run": r, "dataset": cache[r][1], "calls": int(s["calls"]),
                    "api_fail": int(s["api_fail"]), "parse_fail": int(s["parse_fail"]),
                    "calls_tested": int(d["n_calls"].sum()),
                    "failed_tested": int((d["n_calls"] - d["n_ok"]).sum()),
                    "latency_mean_s": lat.mean(), "latency_p50_s": lat.median(),
                    "cost_per_1000_usd": 1000 * float(s["cost_usd"]) / float(s["calls"])})
    ops = pd.DataFrame(ops)

    out = HERE / "final_descriptive"
    out.mkdir(exist_ok=True)
    acc.to_csv(out / "accuracy.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    ops.to_csv(out / "ops.csv", index=False)

    pd.set_option("display.width", 250)
    short = lambda s: s.str.replace("final_", "")  # noqa: E731
    a2 = acc.copy(); a2["run"] = short(a2["run"])
    for k in ("carb_g", "kcal"):
        print(f"\n=== {k} (tested positions)")
        print(a2[a2.target == k].drop(columns=["target"]).round(2).to_string(index=False))
    p2 = pairs.copy(); p2["A"] = short(p2["A"]); p2["B"] = short(p2["B"])
    print("\n=== paired differences, descriptive (Delta = B - A, pp; negative = B more accurate)")
    print(p2.round(2).to_string(index=False))
    o2 = ops.copy(); o2["run"] = short(o2["run"])
    print("\n=== operations")
    print(o2.round(3).to_string(index=False))
    print(f"\nwritten: {out}/accuracy.csv, pairs.csv, ops.csv")


if __name__ == "__main__":
    main()
