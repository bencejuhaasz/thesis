#!/usr/bin/env python3
"""
consistency_pairs.py - consistency of the final round, as pre-registered
(notes/prereg_final_2026-11-12.md): the final runs re-ran the meals of the earlier,
already analysed positions; on those meals the final answer is compared with the answer
of an EARLIER run with the same configuration. Two independent calls per meal, so the
metrics are those of consistency.py with k = 2 repeats. No API calls.
Put it next to consistency.py, significance.py and prereg_tests.py.

    venv/bin/python consistency_pairs.py                     # find the earlier runs itself
    venv/bin/python consistency_pairs.py --pair final_n5k_lite_macros=nutrition5k_macros_n50_s42

Matching (automatic): an earlier run (results/*, not final_*, not *q80*) matches a final
run if it has the same dataset, model label, prompt variant and output mode, and its
manifest has the same model entry (model id and extra settings), the same prompt text,
use_text, image side, DFood view and OmniFood view settings. All candidates are printed;
if several match, the one with the most shared meals is used. CHECK THE PRINTED PAIRS.

Meals: only the excluded (earlier) positions of the final run's seed order, present in
both runs with the same ground truth; repeat 0 of the earlier run.

Output: consistency_pairs/consistency_pairs.csv (one row per final run x target) and
consistency_pairs/pairs_used.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import consistency as C  # noqa: E402
import significance as sig  # noqa: E402
import prereg_tests as P  # noqa: E402

TARGETS = ["kcal", "mass_g", "carb_g", "protein_g", "fat_g"]
ARMS = sorted({r for fam in P.FAMILIES.values() for A, B, _ in fam for r in (A, B)})
CFG_KEYS = ["max_image_side", "dfood_views", "omnifood_views", "omnifood_min_views"]


def manifests(run_dir: Path):
    return [json.loads(m.read_text()) for m in sorted(run_dir.glob("manifest_*.json"))]


def signature(run_dir: Path):
    """(group, config) of a single-group run, or None."""
    res = run_dir / "results.jsonl"
    if not res.exists():
        return None
    df = C.load(res)
    groups = df[C.GROUP].drop_duplicates()
    if len(groups) != 1:
        return None
    g = tuple(groups.iloc[0])
    mans = manifests(run_dir)
    if not mans:
        return None
    m = mans[0]
    ds = g[0]
    model = next((x for x in m.get("models", []) if x.get("label") == g[1]), {})
    dcfg = m.get("datasets", {}).get(ds, {})
    cfg = {k: m.get(k) for k in CFG_KEYS}
    cfg.update({"model": model.get("model"), "extra": json.dumps(model.get("extra"), sort_keys=True),
                "prompt": dcfg.get("prompt"), "use_text": bool(dcfg.get("use_text", False))})
    if ds != "dfood_tj":
        cfg.pop("dfood_views")
    if ds != "omnifood8k":
        cfg.pop("omnifood_views"); cfg.pop("omnifood_min_views")
    return g, cfg


def preds(run_dir: Path, ds: str, view):
    """unit -> (pred dict, truth dict), repeat 0, parsed records only."""
    df = C.load(run_dir / "results.jsonl")
    df = df[(df["repeat"] == 0) & df["parse_ok"].astype(bool)]
    out = {}
    for _, r in df.iterrows():
        out[sig.unit_of(ds, r["sample_id"], view)] = (r["pred"] or {}, r["gt"] or {})
    return out


def excluded_units(final: str, results: Path):
    mans = manifests(results / final)
    g, _ = signature(results / final)
    ds = g[0]
    view = mans[0].get("dfood_views")
    order = []
    for s in mans[0]["datasets"][ds]["sample_ids"]:
        u = sig.unit_of(ds, s, view)
        if u not in order:
            order.append(u)
    lo = P.POS[ds][0]
    return set(order[: lo - 1]), ds, view


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pair", nargs="*", default=[], help="FINAL=EARLIER overrides")
    a = ap.parse_args()
    results = HERE / "results"
    forced = dict(p.split("=", 1) for p in a.pair)

    sigs = {}
    for d in sorted(results.iterdir()):
        if d.is_dir() and not d.name.startswith("final_") and "q80" not in d.name:
            s = signature(d)
            if s:
                sigs[d.name] = s

    rows, used = [], []
    for final in ARMS:
        g, cfg = signature(results / final)
        excl, ds, view = excluded_units(final, results)
        fp = preds(results / final, ds, view)
        cands = [forced[final]] if final in forced else \
            [r for r, (g2, c2) in sigs.items() if g2 == g and c2 == cfg]
        best, best_n = None, 0
        for c in cands:
            ep = preds(results / c, ds, view)   # same DFood view: part of the matched config
            n = sum(1 for u in excl if u in fp and u in ep)
            print(f"{final:30s} candidate {c:45s} shared excluded meals {n}")
            if n > best_n:
                best, best_n = c, n
        if best is None or best_n < 10:
            print(f"{final:30s} -> no usable earlier run\n")
            continue
        ep = preds(results / best, ds, view)
        units = sorted(u for u in excl if u in fp and u in ep)
        used.append({"final": final, "earlier": best, "meals": len(units)})
        print(f"{final:30s} -> {best} ({len(units)} meals)\n")
        for k in TARGETS:
            pairs = [(fp[u][0].get(k), ep[u][0].get(k), fp[u][1].get(k), ep[u][1].get(k)) for u in units]
            pairs = [x for x in pairs if None not in x and np.isclose(x[2], x[3], rtol=1e-6, atol=1e-6)]
            if len(pairs) < 10:
                continue
            Pm = np.array([[x[0], x[1]] for x in pairs], float)
            t = np.array([x[2] for x in pairs], float)
            st = C.target_stats(Pm, t, 0, np.random.default_rng(0))
            rows.append({"final": final, "earlier": best, "dataset": ds, "target": k,
                         **{c: st[c] for c in ("meals", "within_SD", "within_%", "RC", "RC_%",
                                               "ICC", "identical%", "pMAE_r0", "pMAE_r1")}})
    out = HERE / "consistency_pairs"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "consistency_pairs.csv", index=False)
    pd.DataFrame(used).to_csv(out / "pairs_used.csv", index=False)
    pd.set_option("display.width", 250)
    if not df.empty:
        df["final"] = df["final"].str.replace("final_", "")
        print(df.drop(columns=["earlier"]).to_string(index=False))
    print(f"\nwritten: {out}/consistency_pairs.csv, pairs_used.csv")
    print("pMAE_r0 = final run, pMAE_r1 = earlier run, on the same meals")


if __name__ == "__main__":
    main()
