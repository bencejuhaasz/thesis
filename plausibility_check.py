#!/usr/bin/env python3
"""
plausibility_check.py - how many schema-valid answers contain physically implausible
numbers? No API calls; reads the raw model answers (the 'text' field) in results/.

    venv/bin/python plausibility_check.py                 # every results/final_* run
    venv/bin/python plausibility_check.py RUN [RUN ...]   # only these runs

Per item of a parsed answer (estimated_* fields), two rules:
  mass rule    carbohydrate + protein + fat > weight                 (macros heavier than the food)
  energy rule  |4*carb + 4*protein + 9*fat - calories| > 50% of calories, with calories >= 20 kcal
               (Atwater factors; 50% tolerance because fiber, alcohol and rounding
               make the match approximate)
Counted on every parsed answer of the run (all positions, the last record per meal,
like the loader of significance.py). Output: plausibility/plausibility.csv, and the
flagged items in plausibility/flagged_items.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TOL = 0.5
MIN_KCAL = 20.0


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check_run(run: str, results: Path):
    rows = [json.loads(l) for l in (results / run / "results.jsonl").open() if l.strip()]
    last = {}
    for r in rows:
        last[(str(r["sample_id"]), r.get("repeat", 0))] = r
    meals = items = mass_bad = energy_bad = meals_bad = 0
    flagged = []
    for (sid, _), r in last.items():
        if not r.get("parse_ok"):
            continue
        try:
            obj = json.loads(r.get("text") or "")
        except (json.JSONDecodeError, TypeError):
            continue  # parsed via the fallback; raw text is not plain JSON
        meals += 1
        bad_here = False
        for it in obj.get("items", []) or []:
            w, kc = num(it.get("estimated_weight_grams")), num(it.get("estimated_calories"))
            c, p, f = (num(it.get(k)) for k in ("estimated_carbohydrate_grams",
                                                 "estimated_protein_grams", "estimated_fat_grams"))
            if None in (w, kc, c, p, f):
                continue
            items += 1
            m_bad = c + p + f > w
            e_bad = kc >= MIN_KCAL and abs(4 * c + 4 * p + 9 * f - kc) > TOL * kc
            mass_bad += m_bad
            energy_bad += e_bad
            if m_bad or e_bad:
                bad_here = True
                flagged.append({"run": run, "sample_id": sid, "item": it.get("name"),
                                "weight_g": w, "kcal": kc, "carb_g": c, "protein_g": p, "fat_g": f,
                                "kcal_from_macros": 4 * c + 4 * p + 9 * f,
                                "mass_rule": m_bad, "energy_rule": e_bad})
        meals_bad += bad_here
    return ({"run": run, "meals": meals, "items": items,
             "items_mass_rule": mass_bad, "items_energy_rule": energy_bad,
             "meals_with_flag": meals_bad,
             "meals_with_flag_%": 100 * meals_bad / meals if meals else float("nan")}, flagged)


def main():
    results = HERE / "results"
    runs = sys.argv[1:] or sorted(p.name for p in results.glob("final_*") if (p / "results.jsonl").exists())
    summ, flags = [], []
    for run in runs:
        s, f = check_run(run, results)
        summ.append(s)
        flags.extend(f)
    out = HERE / "plausibility"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(summ)
    df.to_csv(out / "plausibility.csv", index=False)
    pd.DataFrame(flags).to_csv(out / "flagged_items.csv", index=False)
    pd.set_option("display.width", 250)
    df["run"] = df["run"].str.replace("final_", "")
    print(df.round(2).to_string(index=False))
    print(f"\nflagged items: {len(flags)}  ->  {out}/flagged_items.csv")


if __name__ == "__main__":
    main()
