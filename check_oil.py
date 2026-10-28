#!/usr/bin/env python3
"""
check_oil.py - is the kcal error the model's, or the labels' cooking oil?
Needs runs with macros (P3_macros). Reads only results/<run>/results.jsonl.

    python check_oil.py RUN [RUN ...]

Idea: oil is pure fat. If the labels count all cooking oil (OmniFood8K: raw
ingredients weighed before cooking), the label FAT is inflated but carbohydrate
and protein are not. So compare separately:
  kcal         model kcal vs label kcal
  nonfat_kcal  4*carb + 4*protein (model) vs the same from the labels
  fat_kcal     9*fat (model) vs 9*fat (labels)
  carb_g, protein_g, fat_g, mass_g
and the per-gram (density) versions of kcal and nonfat_kcal.
Then the kcal bias by tercile of the label's fat share of energy: if the error is
concentrated where the label says "most energy is fat", it is the oil.
Paired: only meals with all values present. pMAE = sum|p-t| / sum t.
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(arg):
    p = Path(arg) if arg.endswith(".jsonl") else Path("results") / arg / "results.jsonl"
    out = []
    for l in open(p):
        if not l.strip():
            continue
        r = json.loads(l)
        if not r.get("parse_ok"):
            continue
        p_, t = r["pred"], r["gt"]
        keys = ("kcal", "carb_g", "protein_g", "fat_g", "mass_g")
        if any(p_.get(k) is None or t.get(k) is None for k in keys):
            continue
        out.append({k: (float(p_[k]), float(t[k])) for k in keys})
    return out


def stats(p, t):
    with np.errstate(all="ignore"):
        return (100 * np.abs(p - t).sum() / t.sum(), 100 * (p.sum() / t.sum() - 1),
                np.corrcoef(p, t)[0, 1])


for arg in sys.argv[1:]:
    rows = load(arg)
    print(f"\n==================== {arg}  ({len(rows)} meals with kcal, macros and mass)")
    if len(rows) < 5:
        print("  too few (run with P3_macros)")
        continue
    A = {k: (np.array([r[k][0] for r in rows]), np.array([r[k][1] for r in rows]))
         for k in rows[0]}
    P = {**A,
         "nonfat_kcal": (4 * A["carb_g"][0] + 4 * A["protein_g"][0], 4 * A["carb_g"][1] + 4 * A["protein_g"][1]),
         "fat_kcal": (9 * A["fat_g"][0], 9 * A["fat_g"][1])}
    P["kcal_per_g"] = (A["kcal"][0] / A["mass_g"][0], A["kcal"][1] / A["mass_g"][1])
    P["nonfat_kcal_per_g"] = (P["nonfat_kcal"][0] / A["mass_g"][0], P["nonfat_kcal"][1] / A["mass_g"][1])
    print(f"  {'target':20s} {'pMAE%':>6s} {'bias%':>7s} {'r':>5s}   label mean")
    for k in ("kcal", "nonfat_kcal", "fat_kcal", "carb_g", "protein_g", "fat_g", "mass_g",
              "kcal_per_g", "nonfat_kcal_per_g"):
        p, t = P[k]
        pm, b, r = stats(p, t)
        print(f"  {k:20s} {pm:6.1f} {b:+7.1f} {r:5.2f}   {t.mean():8.2f}")
    share = P["fat_kcal"][1] / A["kcal"][1]
    edges = np.quantile(share, [0, 1 / 3, 2 / 3, 1])
    print("  kcal and fat bias by the LABEL's fat share of energy (terciles):")
    for i in range(3):
        m = (share >= edges[i]) & (share <= edges[i + 1]) if i == 2 else (share >= edges[i]) & (share < edges[i + 1])
        if m.sum() == 0:
            continue
        pk, tk = A["kcal"][0][m], A["kcal"][1][m]
        pn, tn = P["nonfat_kcal"][0][m], P["nonfat_kcal"][1][m]
        pf, tf = A["fat_g"][0][m], A["fat_g"][1][m]
        print(f"    fat share {100 * edges[i]:3.0f}-{100 * edges[i + 1]:3.0f}%  n={m.sum():3d}   "
              f"kcal bias {100 * (pk.sum() / tk.sum() - 1):+6.1f}%   nonfat bias {100 * (pn.sum() / tn.sum() - 1):+6.1f}%   "
              f"fat bias {100 * (pf.sum() / tf.sum() - 1):+6.1f}%   label fat {tf.mean():5.1f} g/meal")
