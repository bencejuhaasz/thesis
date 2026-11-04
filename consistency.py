#!/usr/bin/env python3
"""
consistency.py - how much does the SAME model give different answers to the SAME photo?
Needs a run with REPEATS >= 2 (benchmark.py --repeats 3). Put it next to benchmark.py.

    venv/bin/python consistency.py --runs nutrition5k_lite_rep3_n50_s42
    venv/bin/python consistency.py --runs RUN --bootstrap 2000      # CI of the averaging gain

Per run x target (meals with >= 2 parsed repeats):
  within_SD       pooled SD of the repeats of one meal (units)       sqrt(mean of per-meal variances)
  within_%        within_SD / mean truth * 100                        same scale as pMAE
  RC              repeatability coefficient = 2.77 * within_SD: two calls on the same photo
                  differ by less than this in 95% of cases (Bland-Altman)
  RC_%            RC / mean truth * 100
  CV_med%         median over meals of SD / mean of that meal's predictions
  ICC             between-meal share of prediction variance (1 = repeats agree perfectly
                  relative to how much meals differ; one-way random, ICC(1,1))
  noise_share%    within-repeat variance / mean squared error of a single call: the part of
                  the error that averaging several calls could remove at most
  identical%      meals where every repeat gave exactly the same value
  pMAE_r0/r1/..   pMAE of each repeat alone (single call)
  pMAE_single     mean of those
  pMAE_avg        pMAE of the per-meal mean of the repeats
  gain_pp         pMAE_avg - pMAE_single (negative = averaging helps) [bootstrap CI over meals]
Item lists: mean pairwise Jaccard of the item names between repeats, SD of the item count.
Output: consistency/consistency.csv
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TARGETS = ["kcal", "mass_g", "carb_g", "protein_g", "fat_g"]
GROUP = ["dataset", "model_label", "prompt_variant", "output_mode"]


def load(res: Path) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in res.open() if l.strip()])
    for c in ("prompt_variant", "output_mode"):
        df[c] = df[c].fillna("?") if c in df else "?"
    if "repeat" not in df:
        df["repeat"] = 0
    return df.drop_duplicates(GROUP + ["sample_id", "repeat"], keep="last")


def pmae(p, t):
    return 100 * np.abs(p - t).sum() / t.sum()


def names(pred):
    out = set()
    for n in (pred or {}).get("item_names") or []:
        if isinstance(n, str) and n.strip():
            out.add(" ".join(n.lower().replace(",", " ").split()))
    return out


def target_stats(P: np.ndarray, t: np.ndarray, boot: int, rng) -> dict:
    """P: meals x repeats (complete rows), t: truth per meal."""
    n, k = P.shape
    tm = t.mean()
    var_w = P.var(axis=1, ddof=1)
    sd_w = math.sqrt(var_w.mean())
    m = P.mean(axis=1)
    ms_b = k * ((m - m.mean()) ** 2).sum() / (n - 1)
    ms_w = var_w.mean()
    icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w) if ms_b + (k - 1) * ms_w > 0 else float("nan")
    mse = ((P - t[:, None]) ** 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(m > 0, np.sqrt(var_w) / m, np.nan)
    singles = [pmae(P[:, j], t) for j in range(k)]
    r = {"meals": n, "repeats": k,
         "within_SD": round(sd_w, 2), "within_%": round(100 * sd_w / tm, 1),
         "RC": round(2.77 * sd_w, 1), "RC_%": round(100 * 2.77 * sd_w / tm, 1),
         "CV_med%": round(100 * np.nanmedian(cv), 1), "ICC": round(icc, 3),
         "noise_share%": round(100 * ms_w / mse, 1) if mse > 0 else float("nan"),
         "identical%": round(100 * np.mean(np.all(P == P[:, :1], axis=1)), 1)}
    for j, s in enumerate(singles):
        r[f"pMAE_r{j}"] = round(s, 1)
    r["pMAE_single"] = round(float(np.mean(singles)), 1)
    r["pMAE_avg"] = round(pmae(m, t), 1)
    r["gain_pp"] = round(r["pMAE_avg"] - float(np.mean(singles)), 2)
    if boot:
        g = []
        for _ in range(boot):
            i = rng.integers(0, n, n)
            Pi, ti = P[i], t[i]
            g.append(pmae(Pi.mean(axis=1), ti) - np.mean([pmae(Pi[:, j], ti) for j in range(k)]))
        r["gain_CI_low"], r["gain_CI_high"] = (round(float(x), 2) for x in np.percentile(g, [2.5, 97.5]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N")
    ap.add_argument("--out", default="consistency")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for run in a.runs:
        res = Path(a.results) / run / "results.jsonl"
        if not res.is_file():
            print(f"skip {run}: no results.jsonl")
            continue
        df = load(res)
        for key, g in df.groupby(GROUP):
            reps = sorted(g["repeat"].unique())
            if len(reps) < 2:
                print(f"skip {run} / {key[1]}: only {len(reps)} repeat(s) - run with --repeats 3")
                continue
            ok = g[g["parse_ok"].astype(bool)]
            fails = len(g) - len(ok)
            print(f"\n{run} | {key[0]} | {key[1]} | {key[2]} | repeats {len(reps)}, calls {len(g)}, "
                  f"failed/unparsed {fails}")
            by = {(str(r["sample_id"]), r["repeat"]): r for _, r in ok.iterrows()}
            sids = sorted({s for s, _ in by})
            jac, cnt = [], []
            for s in sids:
                preds = [by[(s, rp)]["pred"] for rp in reps if (s, rp) in by]
                if len(preds) < 2:
                    continue
                ns = [names(p) for p in preds]
                jac += [len(x & y) / len(x | y) for x, y in itertools.combinations(ns, 2) if x | y]
                cnt.append(np.std([p.get("n_items", 0) for p in preds], ddof=1))
            print(f"  items: mean pairwise Jaccard of names {np.mean(jac):.2f}, "
                  f"SD of item count {np.mean(cnt):.2f} (mean over meals)")
            for tg in TARGETS:
                P, t = [], []
                for s in sids:
                    vals = [(by[(s, rp)]["pred"] or {}).get(tg) if (s, rp) in by else None for rp in reps]
                    tv = (by[(s, reps[0])]["gt"] if (s, reps[0]) in by else
                          next(by[(s, rp)]["gt"] for rp in reps if (s, rp) in by)).get(tg)
                    if tv is None or any(v is None for v in vals):
                        continue
                    P.append([float(v) for v in vals]); t.append(float(tv))
                if len(P) < 5:
                    continue
                r = target_stats(np.array(P), np.array(t), a.bootstrap, rng)
                ci = (f" [{r['gain_CI_low']:+.1f}; {r['gain_CI_high']:+.1f}]" if a.bootstrap else "")
                singles = "/".join(f"{r[f'pMAE_r{j}']:.1f}" for j in range(r["repeats"]))
                print(f"  {tg:10s} within {r['within_%']:5.1f}%  RC {r['RC_%']:5.1f}%  CV_med {r['CV_med%']:4.1f}%  "
                      f"ICC {r['ICC']:.3f}  noise {r['noise_share%']:4.1f}% of MSE  identical {r['identical%']:4.1f}%   "
                      f"pMAE single {singles} -> avg {r['pMAE_avg']:.1f}  gain {r['gain_pp']:+.2f} pp{ci}")
                rows.append({"run": run, "dataset": key[0], "model": key[1], "prompt": key[2],
                             "target": tg, "item_jaccard": round(float(np.mean(jac)), 3),
                             "item_count_SD": round(float(np.mean(cnt)), 2), **r})
    if not rows:
        raise SystemExit("nothing to report")
    out = HERE / a.out
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "consistency.csv", index=False)
    print(f"\n-> {out / 'consistency.csv'}")


if __name__ == "__main__":
    main()
