#!/usr/bin/env python3
"""
prereg_tests.py - the pre-registered hypothesis tests of the final round
(notes/prereg_final_2026-11-12.md). No API calls. Put it next to significance.py;
it reuses significance.py's loader and formulas (delta-method SE, Student t), so the
numbers are computed the same way as in pvalues.py.

    venv/bin/python prereg_tests.py                  # all families
    venv/bin/python prereg_tests.py --families A B   # only these

What it does, per test (exactly as pre-registered):
  - only the TESTED positions of the seed-42 order enter the test (the earlier,
    already analysed positions are excluded); the position of a meal is its place in
    the run manifest's sample_ids list (first manifest of the run; all manifests of a
    run must list the same sample_ids, otherwise the script stops)
  - paired comparison on the meals scored in both arms (missing / unparsed pairs drop
    out and are counted)
  - Delta = pMAE(B) - pMAE(A) in percentage points (negative = B more accurate),
    SE from the per-meal differences (delta method), two-sided Student t, df = n - 1,
    95% CI without correction, Wilcoxon signed-rank p as a robustness check
  - Holm correction within each family; significant if p_holm < 0.05

Output: prereg_tests/prereg_tests.csv and the same table in the terminal.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import significance as sig  # noqa: E402

try:
    from scipy.stats import t as _t, wilcoxon as _wilcoxon
except Exception:
    raise SystemExit("scipy is needed (pip install scipy)")

# tested positions (1-based, inclusive) of the seed-42 order, per dataset
POS = {"nutrition5k": (51, 250), "omnifood8k": (101, 300), "acetada": (51, 250),
       "dfood_tj": (351, 600), "snapme": (51, 250)}
RUNDS = {"n5k": "nutrition5k", "omni": "omnifood8k", "acetada": "acetada",
         "dfood": "dfood_tj", "snapme": "snapme"}

# family -> list of (A run, B run, target); Delta = B - A
FAMILIES = {
    "A": [(f"final_{d}_lite_macros", f"final_{d}_{o}_macros", "carb_g")
          for d in ("n5k", "omni", "acetada", "dfood") for o in ("sonnet", "sol")],
    "B": [("final_snapme_lite_macros", "final_snapme_lite_hint", k) for k in ("carb_g", "fat_g")],
    "C1": [("final_omni_lite_macros", "final_omni_lite_recipe", "fat_g")],
    "C2": [("final_omni_lite_macros", "final_omni_lite_recipe", "carb_g"),
           ("final_n5k_lite_macros", "final_n5k_lite_recipe", "carb_g")],
    "C3": [("final_omni_sol_macros", "final_omni_sol_recipe", "fat_g")],
}


def load_run(run: str, results: Path):
    """Per-meal frame of the run's only (dataset, model, prompt, output) group, restricted
    to the tested positions; returns (frame, dataset, n_tested_positions)."""
    res = results / run / "results.jsonl"
    if not res.exists():
        raise SystemExit(f"{run}: no results.jsonl")
    rows = [json.loads(l) for l in res.open() if l.strip()]
    groups = {(r["dataset"], r["model_label"], r.get("prompt_variant") or "?",
               r.get("output_mode") or "?") for r in rows}
    if len(groups) != 1:
        raise SystemExit(f"{run}: expected one group, found {sorted(groups)}")
    ds, model, prompt, output = groups.pop()
    mans = sorted((results / run).glob("manifest_*.json"))
    if not mans:
        raise SystemExit(f"{run}: no manifest")
    lists = [json.loads(m.read_text())["datasets"][ds]["sample_ids"] for m in mans]
    if any(l != lists[0] for l in lists[1:]):
        raise SystemExit(f"{run}: the manifests list different sample_ids")
    view = sig.manifest_view(res)
    order = []
    for s in lists[0]:
        u = sig.unit_of(ds, s, view)
        if u not in order:
            order.append(u)
    lo, hi = POS[ds]
    tested = set(order[lo - 1: hi])
    d = sig.load_group(res, ds, model, prompt, output)
    if d is None:
        raise SystemExit(f"{run}: no parsed records")
    return d[d["unit"].isin(tested)], ds, len(tested)


def test(dA, dB, k):
    a, b = sig.metric_arrays(dA, k, "pMAE"), sig.metric_arrays(dB, k, "pMAE")
    ua = pd.DataFrame({"x": a[1], "t": a[2]}, index=dA.loc[a[0], "unit"].to_numpy())
    ub = pd.DataFrame({"x": b[1], "t": b[2]}, index=dB.loc[b[0], "unit"].to_numpy())
    ua, ub = ua[~ua.index.duplicated()], ub[~ub.index.duplicated()]
    common = ua.index.intersection(ub.index)
    ua, ub = ua.loc[common], ub.loc[common]
    same = np.isclose(ua["t"], ub["t"], rtol=1e-6, atol=1e-6)
    ua, ub = ua[same], ub[same]
    t = ua["t"].to_numpy()
    n = len(t)
    vA, _, _ = sig.ratio_stats(ua["x"].to_numpy(), t)
    vB, _, _ = sig.ratio_stats(ub["x"].to_numpy(), t)
    diff = ub["x"].to_numpy() - ua["x"].to_numpy()
    dlt, se, _ = sig.ratio_stats(diff, t)
    tc = float(_t.ppf(0.975, n - 1))
    p = float(2 * _t.sf(abs(dlt / se), n - 1)) if se > 0 else float("nan")
    try:
        pw = float(_wilcoxon(diff).pvalue)
    except ValueError:
        pw = float("nan")
    return {"n_pairs": n, "pMAE_A": vA, "pMAE_B": vB, "delta_pp": dlt, "SE": se,
            "CI_lo": dlt - tc * se, "CI_hi": dlt + tc * se, "p": p, "p_wilcoxon": pw,
            "dropped_other_truth": int((~same).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--out", default="prereg_tests")
    a = ap.parse_args()
    results = Path(a.results)
    cache, rows = {}, []
    for fam in a.families:
        for A, B, k in FAMILIES[fam]:
            for r in (A, B):
                if r not in cache:
                    cache[r] = load_run(r, results)
            (dA, ds, npos), (dB, _, _) = cache[A], cache[B]
            res = test(dA, dB, k)
            rows.append({"family": fam, "dataset": ds, "A": A, "B": B, "target": k,
                         "tested_positions": npos, **res})
    df = pd.DataFrame(rows)
    df["m"] = df.groupby("family")["p"].transform("size")
    df["p_holm"] = pd.concat([sig_holm(g["p"]) for _, g in df.groupby("family")])
    df["significant"] = df["p_holm"] < 0.05
    out = HERE / a.out
    out.mkdir(exist_ok=True)
    df.to_csv(out / "prereg_tests.csv", index=False)
    pd.set_option("display.width", 250)
    show = df[["family", "dataset", "B", "target", "tested_positions", "n_pairs", "pMAE_A",
               "pMAE_B", "delta_pp", "CI_lo", "CI_hi", "p", "p_holm", "p_wilcoxon", "m",
               "significant"]].copy()
    show["B"] = show["B"].str.replace("final_", "")
    print("Delta = pMAE(B) - pMAE(A), pp; negative = B more accurate. A = the reference arm:")
    print("  A: Flash-Lite P3_macros   B: P3_macros (snapme: A = macros)   C: A = P3_macros\n")
    print(show.round(3).to_string(index=False))
    print(f"\nwritten: {out / 'prereg_tests.csv'}")


def sig_holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p values within one family."""
    out = pd.Series(np.nan, index=p.index)
    q = p.dropna().sort_values()
    m, running = len(q), 0.0
    for i, (idx, val) in enumerate(q.items()):
        running = max(running, min(1.0, (m - i) * val))
        out[idx] = running
    return out


if __name__ == "__main__":
    main()
