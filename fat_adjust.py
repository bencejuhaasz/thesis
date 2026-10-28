#!/usr/bin/env python3
"""
fat_adjust.py - sensitivity analysis: what if part of the label fat (cooking oil
that is not on the plate) is removed from the labels? NOT a method: the size of
the removal is taken from the models' own fat estimates on the test meals.

    python fat_adjust.py RUN RUN ... [--dataset omnifood8k]

Needs P3_macros runs (kcal + carb/protein/fat). Uses only meals present in all runs.
For every run:  fat_gap  = 1 - sum(model fat) / sum(label fat)          (fraction)
                gap_kcal = 9 * (label fat - model fat), per meal          (kcal)
Adjusted label energy, variants (each with MEAN and MEDIAN over runs):
  prop      kcal - f * 9 * label_fat        f = mean/median fat_gap of ALL runs
  prop_loo  same, f from the OTHER runs only (the run's own fat not used)
  const     kcal - c                        c = mean/median over runs of mean gap_kcal
  meal      kcal - g_i                      g_i = mean/median over runs of meal i's gap
            (strongest leakage: the models' consensus fat replaces the label fat)
Then pMAE, bias, r, skill (vs median guess of the ADJUSTED labels), logSD per run.
Also prints the bias identity: kcal bias = nonfat share * nonfat bias + fat share * fat bias.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(arg, dataset):
    p = Path(arg) if arg.endswith(".jsonl") else Path("results") / arg / "results.jsonl"
    out = {}
    for l in open(p):
        if not l.strip():
            continue
        r = json.loads(l)
        if not r.get("parse_ok") or (dataset and r.get("dataset") != dataset):
            continue
        keys = ("kcal", "carb_g", "protein_g", "fat_g")
        if any(r["pred"].get(k) is None or r["gt"].get(k) is None for k in keys):
            continue
        out[r["sample_id"]] = {k: (float(r["pred"][k]), float(r["gt"][k])) for k in keys}
    return p.parent.name, out


@np.errstate(all="ignore")
def stats(p, t):
    med = np.array([np.median(np.delete(t, i)) for i in range(len(t))])
    pos = (p > 0) & (t > 0)
    return (100 * np.abs(p - t).sum() / t.sum(), 100 * (p.sum() / t.sum() - 1),
            np.corrcoef(p, t)[0, 1], 1 - np.abs(p - t).sum() / np.abs(med - t).sum(),
            np.std(np.log(p[pos] / t[pos])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dataset", default="omnifood8k")
    a = ap.parse_args()
    runs = [load(r, a.dataset) for r in a.runs]
    common = sorted(set.intersection(*(set(d) for _, d in runs)))
    print(f"{len(common)} meals in all {len(runs)} runs")
    if len(common) < 5:
        raise SystemExit("too few shared meals")
    names = [n for n, _ in runs]
    T = {k: np.array([runs[0][1][s][k][1] for s in common]) for k in ("kcal", "carb_g", "protein_g", "fat_g")}
    P = [{k: np.array([d[s][k][0] for s in common]) for k in T} for _, d in runs]

    # --- bias identity
    tn, tf = 4 * T["carb_g"] + 4 * T["protein_g"], 9 * T["fat_g"]
    sn, sf = tn.sum() / T["kcal"].sum(), tf.sum() / T["kcal"].sum()
    print(f"\nlabel energy shares: non-fat {100 * sn:.1f}%, fat {100 * sf:.1f}%  (sum {100 * (sn + sf):.1f}%, Atwater)")
    print(f"{'run':45s} {'kcal bias':>9s} {'nonfat':>8s} {'fat':>8s} {'identity':>9s}")
    gaps, gapk = [], []
    for n, p in zip(names, P):
        pn, pf = 4 * p["carb_g"] + 4 * p["protein_g"], 9 * p["fat_g"]
        bk = 100 * (p["kcal"].sum() / T["kcal"].sum() - 1)
        bn = 100 * (pn.sum() / tn.sum() - 1)
        bf = 100 * (pf.sum() / tf.sum() - 1)
        print(f"{n[:45]:45s} {bk:+8.1f}% {bn:+7.1f}% {bf:+7.1f}% {sn * bn + sf * bf:+8.1f}%")
        gaps.append(1 - p["fat_g"].sum() / T["fat_g"].sum())
        gapk.append(9 * (T["fat_g"] - p["fat_g"]))
    gaps, gapk = np.array(gaps), np.array(gapk)          # gapk: runs x meals
    print("fat_gap per run: " + "  ".join(f"{g:.3f}" for g in gaps)
          + f"   mean {gaps.mean():.3f}  median {np.median(gaps):.3f}")

    # --- adjusted labels
    def variants(i):
        others = np.delete(np.arange(len(runs)), i)
        out = {"original": T["kcal"]}
        for agg_name, agg in (("mean", np.mean), ("median", np.median)):
            out[f"prop_{agg_name}"] = T["kcal"] - agg(gaps) * tf
            if len(others):
                out[f"prop_loo_{agg_name}"] = T["kcal"] - agg(gaps[others]) * tf
            out[f"const_{agg_name}"] = T["kcal"] - agg(gapk.mean(axis=1))
            out[f"meal_{agg_name}"] = T["kcal"] - agg(gapk, axis=0)
        return {k: np.clip(v, 1.0, None) for k, v in out.items()}

    keys = list(variants(0))
    for key in keys:
        print(f"\n=== labels: {key}" + ("   (leakage: models' consensus fat = truth)" if key.startswith("meal") else ""))
        print(f"  {'run':45s} {'pMAE%':>6s} {'bias%':>7s} {'r':>5s} {'skill':>6s} {'logSD':>6s}  label mean")
        for i, (n, p) in enumerate(zip(names, P)):
            t = variants(i)[key]
            pm, b, r, sk, ls = stats(p["kcal"], t)
            print(f"  {n[:45]:45s} {pm:6.1f} {b:+7.1f} {r:5.2f} {sk:6.2f} {ls:6.3f}  {t.mean():8.1f}")


if __name__ == "__main__":
    main()
