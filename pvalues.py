#!/usr/bin/env python3
"""
pvalues.py - p values for every pair of runs on the same dataset, from results/.

Put it next to benchmark.py AND significance.py (it reuses significance.py's loader
and formulas, so both give the same numbers). No API calls.

    python pvalues.py                                   # every results/*/summary.csv
    python pvalues.py --exclude smoke dry pre           # skip runs whose name contains these
    python pvalues.py --runs runA runB runC             # only these runs
    python pvalues.py --metrics kcal_pMAE% carb_g_pMAE% # default: every pMAE metric
    python pvalues.py --bias                            # also the bias% metrics

For every dataset, every pair of runs, every metric:
    paired   (both runs scored the same meals, >= --min-common shared):
             Delta = B - A on the shared meals, SE from the per-meal differences,
             Student t test (df = n - 1)  +  Wilcoxon signed-rank p (no normality
             assumption; needs scipy) as a robustness check
    unpaired (different meals): Delta of the two runs' own values,
             SE = sqrt(SE_A^2 + SE_B^2), normal/t approximation. Weaker: prefer paired.
    95% CI of Delta, and Holm-corrected p within each family (default: one family =
    one dataset x one metric, i.e. all pairs compared on that metric).
    pMAE: Delta < 0 means B is more accurate.  bias%: Delta = change of the bias.

Output (--out, default pvalues/):
    pvalues.csv   one row per dataset x metric x pair, sorted by p
    The terminal shows the same, grouped, with * p<0.05 ** p<0.01 *** p<0.001
    (stars on the Holm p; '.' = p<0.1).
Run names, not sheet rows: the 'label' column (model | prompt | images per call | n)
helps to find the matching row in your sheet.
"""
from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import significance as sig
except ImportError:
    raise SystemExit("significance.py must be in the same folder as pvalues.py")

try:
    from scipy.stats import wilcoxon as _wilcoxon
except Exception:
    _wilcoxon = None


def align(dA, dB, k, kind, min_common):
    """Per-meal x (|e| or e) of both runs on the shared meals with identical truth."""
    a, b = sig.metric_arrays(dA, k, kind), sig.metric_arrays(dB, k, kind)
    if a is None or b is None:
        return None
    ia, xa, ta = a
    ib, xb, tb = b
    ua = pd.DataFrame({"x": xa, "t": ta}, index=dA.loc[ia, "unit"].to_numpy())
    ub = pd.DataFrame({"x": xb, "t": tb}, index=dB.loc[ib, "unit"].to_numpy())
    ua, ub = ua[~ua.index.duplicated()], ub[~ub.index.duplicated()]
    common = ua.index.intersection(ub.index)
    if len(common) < min_common:
        return None
    ua, ub = ua.loc[common], ub.loc[common]
    same = np.isclose(ua["t"], ub["t"], rtol=1e-6, atol=1e-6)
    ua, ub = ua[same], ub[same]
    if len(ua) < min_common:
        return None
    return ua["x"].to_numpy(), ub["x"].to_numpy(), ua["t"].to_numpy()


def t_crit(n):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, max(n - 1, 1)))
    except Exception:
        return 1.96


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p values (NaN kept)."""
    out = pd.Series(np.nan, index=p.index)
    q = p.dropna().sort_values()
    m, running = len(q), 0.0
    for i, (idx, val) in enumerate(q.items()):
        running = max(running, min(1.0, (m - i) * val))
        out[idx] = running
    return out


def label_of(row, n):
    ipc = row.get("images_per_call", "")
    ipc = f"{float(ipc):g} img" if str(ipc).strip() not in ("", "nan") else "? img"
    return f"{row['model']} | {row['prompt']} | {ipc} | n={n}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--runs", nargs="+", help="only these run names")
    ap.add_argument("--exclude", nargs="+", default=[], help="skip runs whose name contains any of these")
    ap.add_argument("--metrics", nargs="+", help="e.g. kcal_pMAE%% mass_g_pMAE%% carb_g_bias%%")
    ap.add_argument("--bias", action="store_true", help="also the bias%% metrics")
    ap.add_argument("--min-common", type=int, default=10, help="shared meals for a paired test")
    ap.add_argument("--min-n", type=int, default=10, help="skip runs with fewer meals")
    ap.add_argument("--no-unpaired", action="store_true", help="only paired comparisons")
    ap.add_argument("--family", default="dataset_metric", choices=["dataset_metric", "dataset", "all"],
                    help="which tests share one Holm correction")
    ap.add_argument("--out", default="pvalues")
    a = ap.parse_args()

    res_dir = Path(a.results)
    files = sorted(res_dir.glob("*/summary.csv"))
    if a.runs:
        files = [res_dir / r / "summary.csv" for r in a.runs]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"not found: {[str(m) for m in missing]}")
    files = [f for f in files if not any(x in f.parent.name for x in a.exclude)]
    if not files:
        raise SystemExit(f"no summary.csv under {res_dir}")
    summ = sig.read_summaries([str(f) for f in files])

    wanted = set(a.metrics) if a.metrics else None
    metrics = [(k, kind) for k, kind in sig.METRICS
               if (wanted and sig.mname(k, kind) in wanted)
               or (not wanted and (kind == "pMAE" or a.bias))]

    data, info = {}, {}
    for i, row in summ.iterrows():
        res = sig.results_path(row["run"], row["_source"], res_dir)
        if res is None:
            print(f"  skip {row['run']}: no results.jsonl")
            continue
        d = sig.load_group(res, row["dataset"], row["model"], row["prompt"], row.get("output", "?") or "?")
        if d is None or len(d) < a.min_n:
            print(f"  skip {row['run']} / {row['dataset']} / {row['model']} / {row['prompt']}: "
                  + ("no records" if d is None else f"only {len(d)} meals"))
            continue
        data[i] = d
        info[i] = {"run": row["run"], "dataset": row["dataset"], "label": label_of(row, len(d))}
    print(f"{len(data)} run rows loaded, p values: {sig.P_METHOD}"
          + ("" if _wilcoxon else "; Wilcoxon needs scipy") + "\n")

    rows = []
    for ka, kb in combinations(list(data), 2):
        A, B = info[ka], info[kb]
        if A["dataset"] != B["dataset"]:
            continue
        for k, kind in metrics:
            mn = sig.mname(k, kind)
            al = align(data[ka], data[kb], k, kind, a.min_common)
            rec = {"dataset": A["dataset"], "metric": mn, "run_A": A["run"], "run_B": B["run"],
                   "label_A": A["label"], "label_B": B["label"]}
            if al is not None:
                xa, xb, t = al
                n = len(t)
                vA, _, _ = sig.ratio_stats(xa, t)
                vB, _, _ = sig.ratio_stats(xb, t)
                dlt, se, _ = sig.ratio_stats(xb - xa, t)
                z = dlt / se if se > 0 else (0.0 if abs(dlt) < 1e-9 else float("nan"))
                p = sig.p_two_sided(z, n) if z == z else float("nan")
                pw = float("nan")
                if _wilcoxon is not None and np.any(xb - xa != 0):
                    try:
                        pw = float(_wilcoxon(xb, xa).pvalue)
                    except Exception:
                        pass
                rec.update(test="paired", n=n, A=vA, B=vB, delta_pp=dlt, SE=se, p=p, p_wilcoxon=pw)
            elif not a.no_unpaired:
                ma, mb = sig.metric_arrays(data[ka], k, kind), sig.metric_arrays(data[kb], k, kind)
                if ma is None or mb is None:
                    continue
                vA, seA, _ = sig.ratio_stats(ma[1], ma[2])
                vB, seB, _ = sig.ratio_stats(mb[1], mb[2])
                n = min(len(ma[2]), len(mb[2]))
                dlt, se = vB - vA, math.sqrt(seA ** 2 + seB ** 2)
                z = dlt / se if se > 0 else float("nan")
                rec.update(test="unpaired", n=n, A=vA, B=vB, delta_pp=dlt, SE=se,
                           p=sig.p_two_sided(z, n) if z == z else float("nan"), p_wilcoxon=float("nan"))
            else:
                continue
            tc = t_crit(rec["n"])
            rec["CI95_low"], rec["CI95_high"] = rec["delta_pp"] - tc * rec["SE"], rec["delta_pp"] + tc * rec["SE"]
            rows.append(rec)

    if not rows:
        raise SystemExit("no comparable pairs (same dataset, same metric)")
    df = pd.DataFrame(rows)
    fam = {"dataset_metric": ["dataset", "metric"], "dataset": ["dataset"], "all": None}[a.family]
    df["p_holm"] = holm(df["p"]) if fam is None else df.groupby(fam, group_keys=False)["p"].apply(holm)
    df["family_size"] = len(df) if fam is None else df.groupby(fam)["p"].transform("count")
    df["sig_raw"] = df["p"].map(sig.stars)
    df["sig_holm"] = df["p_holm"].map(sig.stars)
    df = df.sort_values(["dataset", "metric", "p"]).reset_index(drop=True)
    num = ["A", "B", "delta_pp", "SE", "CI95_low", "CI95_high"]
    df[num] = df[num].round(2)
    for c in ("p", "p_wilcoxon", "p_holm"):
        df[c] = df[c].map(lambda v: float(f"{v:.3g}") if v == v else v)
    cols = ["dataset", "metric", "test", "n", "run_A", "label_A", "run_B", "label_B", "A", "B",
            "delta_pp", "CI95_low", "CI95_high", "SE", "p", "p_wilcoxon", "p_holm", "family_size",
            "sig_raw", "sig_holm"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out / "pvalues.csv", index=False)

    for (ds, mn), g in df.groupby(["dataset", "metric"], sort=False):
        print(f"=== {ds}  {mn}   ({len(g)} comparisons, Holm within this group)"
              if a.family == "dataset_metric" else f"=== {ds}  {mn}")
        for _, r in g.iterrows():
            w = f"  W {r['p_wilcoxon']:.3g}" if r["p_wilcoxon"] == r["p_wilcoxon"] else ""
            print(f"  {r['label_A'][:44]:44s} -> {r['label_B'][:44]:44s} {r['test'][:4]:4s} "
                  f"{r['A']:6.1f} -> {r['B']:6.1f}  Δ {r['delta_pp']:+6.1f} "
                  f"[{r['CI95_low']:+.1f}; {r['CI95_high']:+.1f}]  p {r['p']:<7.3g} Holm {r['p_holm']:<7.3g}"
                  f"{r['sig_holm']:3s}{w}")
            print(f"      {r['run_A']}  ->  {r['run_B']}")
        print()
    print(f"-> {out / 'pvalues.csv'}   ({len(df)} comparisons; "
          f"{int((df['p_holm'] < 0.05).sum())} with Holm p < 0.05, {int((df['p'] < 0.05).sum())} with raw p < 0.05)")


if __name__ == "__main__":
    main()
