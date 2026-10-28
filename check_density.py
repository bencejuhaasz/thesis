#!/usr/bin/env python3
"""
check_density.py - independent check of the kcal/g numbers (does not import
benchmark.py or analyze.py; reads only results/<run>/results.jsonl).

    python check_density.py RUN [RUN ...]

1. PARSING   re-parses every raw model answer ("text") and re-sums kcal and grams;
             compares with the stored pred (mismatches are printed).
2. LABELS    true kcal/g per meal; Atwater check of the labels
             (4*carb + 4*protein + 9*fat) / kcal - about 1.0 if kcal is really kcal
             (about 0.24 would mean kJ); share of meals with implausible kcal/g.
3. ITEMS     the model's own kcal/100 g per item, for the most frequent item names,
             so you can judge them against what you know (cooked rice ~110-140,
             leafy veg 15-40, stir-fried veg 50-150, fatty meat 250-500 ...).
4. MEALS     a few meals in full: true kcal, g, kcal/g vs the model's items.
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def runfile(arg):
    p = Path(arg)
    if p.suffix == ".jsonl":
        return p
    return Path("results") / arg / "results.jsonl"


def parse_raw(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        return json.loads(m.group(0)) if m else None


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


for arg in sys.argv[1:]:
    recs = [json.loads(l) for l in open(runfile(arg)) if l.strip()]
    recs = [r for r in recs if r.get("parse_ok")]
    print(f"\n==================== {arg}  ({len(recs)} parsed)")

    # 1. parsing
    bad, items_all, meals = 0, defaultdict(list), []
    for r in recs:
        obj = parse_raw(r.get("text"))
        its = obj.get("items", []) if isinstance(obj, dict) else []
        k = sum(float(i.get("estimated_calories", 0)) for i in its)
        g = sum(float(i.get("estimated_weight_grams", 0)) for i in its)
        pk, pg = r["pred"].get("kcal"), r["pred"].get("mass_g")
        if pk is None or abs(k - pk) > 0.01 or abs(g - (pg or 0)) > 0.01:
            bad += 1
            print(f"  MISMATCH {r['sample_id']}: raw kcal {k} g {g} vs stored {pk} {pg}")
        for i in its:
            gi = float(i.get("estimated_weight_grams", 0))
            if gi > 0:
                items_all[str(i.get("name", "")).lower().strip()].append(
                    100 * float(i.get("estimated_calories", 0)) / gi)
        meals.append((r, its, k, g))
    print(f"1. PARSING: {len(recs) - bad}/{len(recs)} stored predictions equal the raw answer"
          + ("" if bad == 0 else "   <-- CHECK"))

    # 2. labels
    tk = [(r, r["gt"]["kcal"], r["gt"].get("mass_g")) for r, *_ in meals
          if r["gt"].get("kcal") and r["gt"].get("mass_g")]
    dens = [k / g for _, k, g in tk]
    atw = [(4 * r["gt"]["carb_g"] + 4 * r["gt"]["protein_g"] + 9 * r["gt"]["fat_g"]) / r["gt"]["kcal"]
           for r, *_ in meals if all(r["gt"].get(x) is not None for x in ("carb_g", "protein_g", "fat_g"))
           and r["gt"].get("kcal")]
    print(f"2. LABELS: true kcal/g median {st.median(dens):.2f}  (10%..90%: {q(dens, .1):.2f}..{q(dens, .9):.2f}),"
          f"  >4 kcal/g: {sum(d > 4 for d in dens)},  <0.3 kcal/g: {sum(d < 0.3 for d in dens)}")
    if atw:
        print(f"   Atwater (4C+4P+9F)/kcal median {st.median(atw):.2f}  (10%..90%: {q(atw, .1):.2f}..{q(atw, .9):.2f})"
              "   ~1.0 = kcal labels consistent with the macro labels")
    fat_share = [9 * r["gt"]["fat_g"] / r["gt"]["kcal"] for r, *_ in meals
                 if r["gt"].get("fat_g") is not None and r["gt"].get("kcal")]
    if fat_share:
        print(f"   share of energy from fat (labels): median {100 * st.median(fat_share):.0f}%")
    pd_ = [k / g for _, _, k, g in meals if g > 0]
    print(f"   model kcal/g median {st.median(pd_):.2f}  (10%..90%: {q(pd_, .1):.2f}..{q(pd_, .9):.2f})")

    # 3. items
    print("3. ITEMS: model's kcal/100 g, most frequent names (n, median, min..max)")
    top = sorted(items_all.items(), key=lambda kv: -len(kv[1]))[:15]
    for name, v in top:
        print(f"   {name[:38]:38s} n={len(v):3d}  {st.median(v):5.0f}  ({min(v):.0f}..{max(v):.0f})")

    # 4. a few meals in full (largest kcal underestimates)
    print("4. MEALS with the largest kcal underestimate:")
    worst = sorted(meals, key=lambda m: m[2] / m[0]["gt"]["kcal"] if m[0]["gt"].get("kcal") else 9)[:3]
    for r, its, k, g in worst:
        gt = r["gt"]
        print(f"   {r['sample_id']}: TRUE {gt['kcal']:.0f} kcal, {gt.get('mass_g') or 0:.0f} g, "
              f"{gt['kcal'] / gt['mass_g'] if gt.get('mass_g') else float('nan'):.2f} kcal/g, fat {gt.get('fat_g')} g"
              f"  |  MODEL {k:.0f} kcal, {g:.0f} g, {k / g if g else float('nan'):.2f} kcal/g")
        for i in its:
            gi = float(i.get("estimated_weight_grams", 0))
            ki = float(i.get("estimated_calories", 0))
            print(f"      - {str(i.get('name'))[:40]:40s} {gi:5.0f} g {ki:5.0f} kcal "
                  f"({100 * ki / gi if gi else 0:.0f}/100 g)  fat {i.get('estimated_fat_grams')}")
