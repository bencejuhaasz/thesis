#!/usr/bin/env python3
"""
analyze.py - offline analysis of benchmark runs (no API calls).

Put it next to benchmark.py (in the folder that contains the dataset folders).

    python analyze.py compare   BASELINE OTHER ... [--target carb_g] [--calibrate]
    python analyze.py decompose RUN ... [--target carb_g] [--only-ids RUN]
    python analyze.py spread    [--targets kcal carb_g mass_g]
    python analyze.py sameset   RUN RUN ... [--contains RUN]

A RUN / spec is   NAME[:FIELD][@VARIANT]
    NAME     run name (results/<NAME>/results.jsonl) or a path to a results.jsonl
    FIELD    prediction key to score (default = --target)
    VARIANT  only records of this prompt variant (if a run holds several)
--dataset limits every command to one dataset (needed if a run holds several).

compare    paired comparison on the meals present in EVERY spec: pMAE, MAE, bias,
           r, slope, skill vs the leave-one-out median guess, logSD (spread of
           ln(pred/true), i.e. error without bias); delta vs the first spec with a
           paired bootstrap 95% CI and a Wilcoxon p. --calibrate removes each
           spec's overall level first (leave-one-out scale factor).
decompose  target = grams x (target per gram); splits the log error into portion
           and density, with the counterfactuals "grams exact" / "density exact".
           Needs true mass: taken from the records (gt.mass_g); for old DFood-TJ
           runs without it, from DFood-Tj/loader.py.
spread     how much each dataset's ground truth varies (median-guess pMAE = the
           score a model must beat); uses benchmark.py's loaders/colmap/query.
sameset    did several runs use the same meals? (DFood-TJ "613+614"/"613"/"614"
           are mapped to one meal; other datasets compare sample_ids directly)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


# ======================================================================
# shared helpers
# ======================================================================

def parse_spec(spec: str, default_field: str):
    """NAME[:FIELD][@VARIANT] -> (path to results.jsonl, field, variant)."""
    variant = None
    if "@" in spec:
        spec, variant = spec.rsplit("@", 1)
    field = default_field
    p = Path(spec)
    if not (p.suffix == ".jsonl" and p.exists()) and ":" in spec:
        spec, field = spec.rsplit(":", 1)
        p = Path(spec)
    if p.is_dir():
        p = p / "results.jsonl"
    if not p.exists():
        p = HERE / "results" / spec / "results.jsonl"
    if not p.exists():
        raise SystemExit(f"{spec}: no results.jsonl found (run name or path)")
    return p, field, variant


def read_records(path: Path, dataset=None, variant=None) -> list[dict]:
    recs = [json.loads(l) for l in path.open() if l.strip()]
    return [r for r in recs if r.get("parse_ok")
            and (dataset is None or r.get("dataset") == dataset)
            and (variant is None or r.get("prompt_variant") == variant)]


def pmae(p, t):
    return 100 * np.abs(p - t).sum() / t.sum()


def run_label(path: Path, field: str, variant: str | None) -> str:
    return f"{path.parent.name}:{field}" + (f"@{variant}" if variant else "")


# ======================================================================
# compare
# ======================================================================

def _compare_load(spec: str, target: str, dataset: str | None):
    p, field, variant = parse_spec(spec, target)
    recs = read_records(p, dataset, variant)
    groups = {(r["dataset"], r["model_label"], r.get("prompt_variant")) for r in recs}
    if len(groups) > 1:
        raise SystemExit(f"{spec}: several dataset/model/variant groups {sorted(groups)}; "
                         "use --dataset and/or @VARIANT")
    by = {}
    for r in recs:                           # repeats -> mean prediction
        pv, tv = r["pred"].get(field), (r.get("gt") or {}).get(target)
        if pv is None or tv is None:
            continue
        by.setdefault(r["sample_id"], ([], tv))[0].append(float(pv))
    vals = {k: (float(np.mean(ps)), t) for k, (ps, t) in by.items()}
    return run_label(p, field, variant), vals


@np.errstate(all="ignore")
def _stats(p, t):
    e = p - t
    loo_med = np.array([np.median(np.delete(t, i)) for i in range(len(t))])
    return {"pMAE%": pmae(p, t), "MAE": np.abs(e).mean(),
            "bias%": 100 * e.sum() / t.sum(), "r": np.corrcoef(p, t)[0, 1],
            "slope": np.polyfit(t, p, 1)[0],
            "skill": 1 - np.abs(e).sum() / np.abs(loo_med - t).sum(),
            "logSD": np.std(np.log(p / t)) if (p > 0).all() and (t > 0).all() else np.nan}


def _loo_scale(p, t):
    """Pure level correction: each meal x (sum true / sum pred) of the OTHER meals."""
    return np.array([p[i] * (t.sum() - t[i]) / (p.sum() - p[i]) for i in range(len(p))])


def cmd_compare(a):
    runs = [_compare_load(s, a.target, a.dataset) for s in a.specs]
    common = sorted(set.intersection(*(set(v) for _, v in runs)))
    print(f"target {a.target}: {len(common)} meals in all {len(runs)} specs "
          f"(per spec: {[len(v) for _, v in runs]})")
    if len(common) < 5:
        raise SystemExit("too few shared meals")
    t = np.array([runs[0][1][k][1] for k in common])
    for lab, v in runs[1:]:
        if not np.allclose(t, [v[k][1] for k in common], rtol=1e-6, atol=1e-6):
            print(f"WARNING: ground truth differs between specs ({lab}) - not the same meals?")
    P = [np.array([v[k][0] for k in common]) for _, v in runs]
    if a.calibrate:
        P = [_loo_scale(p, t) for p in P]
        print("CALIBRATED: every spec rescaled by a leave-one-out level factor")
    print(f"mean truth {t.mean():.1f}   (logSD = SD of ln(pred/true): spread without bias)\n")
    print(f"{'spec':60s} {'pMAE%':>6s} {'MAE':>7s} {'bias%':>6s} {'r':>5s} "
          f"{'slope':>5s} {'skill':>6s} {'logSD':>6s}")
    for (lab, _), p in zip(runs, P):
        s = _stats(p, t)
        print(f"{lab[:60]:60s} {s['pMAE%']:6.1f} {s['MAE']:7.1f} {s['bias%']:+6.1f} "
              f"{s['r']:5.2f} {s['slope']:5.2f} {s['skill']:6.2f} {s['logSD']:6.3f}")
    if len(runs) < 2:
        return
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(t), size=(a.boot, len(t)))
    base_e = np.abs(P[0] - t)
    print(f"\npaired vs {runs[0][0]} (negative = better):")
    for (lab, _), p in zip(runs[1:], P[1:]):
        e = np.abs(p - t)
        d = 100 * (e.sum() - base_e.sum()) / t.sum()
        db = 100 * (e[idx].sum(1) - base_e[idx].sum(1)) / t[idx].sum(1)
        lo, hi = np.percentile(db, [2.5, 97.5])
        try:
            from scipy.stats import wilcoxon
            pv = f"{wilcoxon(e, base_e).pvalue:.2g}"
        except Exception:
            pv = "n/a (pip install scipy)"
        print(f"  {lab[:58]:58s} Δ pMAE {d:+5.1f} pp  CI [{lo:+.1f}, {hi:+.1f}]  "
              f"Wilcoxon p {pv}  meals better {100 * np.mean(e < base_e):.0f}%")


# ======================================================================
# decompose
# ======================================================================

_DFOOD_TRUTH = None


def _dfood_truth(root: Path) -> pd.DataFrame:
    """True mass per DFood-TJ image, for old runs whose records lack gt.mass_g."""
    global _DFOOD_TRUTH
    if _DFOOD_TRUTH is None:
        spec = importlib.util.spec_from_file_location("dfood_loader", root / "loader.py")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(root))
        spec.loader.exec_module(mod)
        images, _ = mod.load_fnutr(str(root))
        if "mass_g" not in images:
            raise SystemExit("DFood loader has no mass_g - use the updated DFood-Tj/loader.py")
        t = images[["image_id", "mass_g"]].copy()
        t["key"] = t["image_id"].map(lambda v: str(int(float(v))))
        _DFOOD_TRUTH = t.set_index("key")["mass_g"]
    return _DFOOD_TRUTH


def _true_mass(r: dict, dfood_root: Path):
    g = (r.get("gt") or {}).get("mass_g")
    if g is not None:
        return g
    if r.get("dataset") == "dfood_tj":
        key = str(int(str(r["sample_id"]).split("#")[0].split("+")[0]))
        v = _dfood_truth(dfood_root).get(key)
        return None if v is None or pd.isna(v) else float(v)
    return None


def _decompose_one(name: str, d: pd.DataFrame, target: str, out_csv: Path):
    print(f"\n==================== {name}  ({len(d)} parsed calls)")
    bad = d[d["true_g"].isna()]
    d = d[d["true_g"].notna()]
    zero = d[(d.pred_g <= 0) | (d.pred_t <= 0) | d.pred_g.isna() | d.pred_t.isna() | (d.true_t <= 0)]
    d = d.drop(zero.index)
    print(f"used {len(d)}; skipped {len(bad)} without true mass, "
          f"{len(zero)} with zero/missing values")
    if len(d) < 5:
        return
    pk, pg, tk, tg = (d[c].to_numpy(float) for c in ("pred_t", "pred_g", "true_t", "true_g"))
    rel = pg / tg - 1
    print(f"\nMASS   MAE {np.abs(pg - tg).mean():.1f} g   pMAE {pmae(pg, tg):.1f}%   "
          f"bias {100 * (pg - tg).sum() / tg.sum():+.1f}%   r {np.corrcoef(pg, tg)[0, 1]:.2f}   "
          f"slope {np.polyfit(tg, pg, 1)[0]:.2f}")
    print("       within ±2% {:.0f}%   ±5% {:.0f}%   ±10% {:.0f}%   ±25% {:.0f}%   "
          "(copying a visible scale -> spike at ±2-5%)".format(
              *(100 * np.mean(np.abs(rel) <= x) for x in (.02, .05, .10, .25))))
    print(f"{target.upper()} pMAE {pmae(pk, tk):.1f}%   bias {100 * (pk - tk).sum() / tk.sum():+.1f}%")
    pdn, tdn = pk / pg, tk / tg
    print(f"DENSITY {target}/g  pred median {np.median(pdn):.3f}   true median {np.median(tdn):.3f}   "
          f"r(pred, true) {np.corrcoef(pdn, tdn)[0, 1]:.2f}")

    rE, rM = np.log(pk / tk), np.log(pg / tg)
    rD = rE - rM
    cov = np.cov(rM, rD, ddof=0)[0, 1]
    print(f"\nlog split   mean (bias)   SD     mean square")
    for lab, x in ((target, rE), ("portion", rM), ("density", rD)):
        print(f"  {lab:9s} {100 * (np.exp(x.mean()) - 1):+8.1f}%   {x.std():.3f}   {np.mean(x ** 2):.4f}")
    print(f"  Var({target}) {rE.var():.4f} = Var(portion) {rM.var():.4f} + "
          f"Var(density) {rD.var():.4f} + 2Cov {2 * cov:+.4f}")
    print(f"\nIF grams were exact   (model density x true g):    {target} pMAE {pmae(pdn * tg, tk):.1f}%")
    print(f"IF density were exact (model g x true {target}/g): {target} pMAE {pmae(pg * tdn, tk):.1f}%")
    print(f"(median guess on this subset: {pmae(np.full(len(tk), np.median(tk)), tk):.1f}%)")
    d.assign(log_target=rE, log_portion=rM, log_density=rD).to_csv(out_csv, index=False)
    print(f"-> {out_csv}")


def cmd_decompose(a):
    only = None
    if a.only_ids:
        p, _, v = parse_spec(a.only_ids, a.target)
        only = {r["sample_id"] for r in read_records(p, a.dataset, v)}
        print(f"restricted to the {len(only)} sample_ids of {p.parent.name}")
    for spec in a.specs:
        p, field, variant = parse_spec(spec, a.target)
        recs = [r for r in read_records(p, a.dataset, variant)
                if only is None or r["sample_id"] in only]
        rows = [dict(dataset=r["dataset"], model=r["model_label"],
                     variant=r.get("prompt_variant"), sample_id=r["sample_id"],
                     repeat=r.get("repeat", 0),
                     pred_t=r["pred"].get(field), pred_g=r["pred"].get("mass_g"),
                     true_t=(r.get("gt") or {}).get(a.target),
                     true_g=_true_mass(r, HERE / a.dfood_root))
                for r in recs]
        d = pd.DataFrame(rows)
        if d.empty:
            print(f"\n{p.parent.name}: no parsed records")
            continue
        d[["pred_t", "pred_g", "true_t", "true_g"]] = d[["pred_t", "pred_g", "true_t", "true_g"]].astype(float)
        for (ds, model, var), g in d.groupby(["dataset", "model", "variant"], dropna=False):
            name = f"{p.parent.name} | {ds} | {model} | {var} | {field}"
            out = HERE / "results" / f"decompose_{p.parent.name}_{ds}_{field}.csv"
            _decompose_one(name, g.reset_index(drop=True), a.target, out)


# ======================================================================
# spread
# ======================================================================

def _spread(v: pd.Series) -> dict | None:
    t = pd.to_numeric(v, errors="coerce").dropna()
    t = t[t > 0].to_numpy(float)
    if len(t) < 5:
        return None
    med = np.median(t)
    q1, q3 = np.percentile(t, [25, 75])
    return {"n": len(t), "mean": t.mean(), "median": med, "CV": t.std() / t.mean(),
            "MG_pMAE%": pmae(np.full(len(t), med), t), "IQR/med": (q3 - q1) / med}


def cmd_spread(a):
    spec = importlib.util.spec_from_file_location("bench", HERE / a.bench)
    b = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE))
    spec.loader.exec_module(b)
    rows = []
    for name, cfg in b.DATASETS.items():
        if a.dataset and name != a.dataset:
            continue
        try:
            root, _, out = b.call_loader(name, cfg)
            pool = b.standardize(name, cfg, b.pick_frame(name, cfg, out), root)
            sel = b.select_rows(pool, cfg.get("select", {"mode": "all"}))
        except Exception as e:
            print(f"[{name}] skipped: {type(e).__name__}: {str(e)[:120]}")
            continue
        for tgt in a.targets:
            for which, df in (("pool", pool), ("selection", sel)):
                s = _spread(df[tgt]) if tgt in df else None
                if s:
                    rows.append({"dataset": name, "target": tgt, "set": which, **s})
                elif which == "pool":
                    print(f"[{name}] no {tgt} ground truth")
    if not rows:
        raise SystemExit("nothing to report")
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for tgt in a.targets:
        for which in ("pool", "selection"):
            t = d[(d.target == tgt) & (d.set == which)].sort_values("MG_pMAE%", ascending=False)
            if t.empty:
                continue
            print(f"\n=== {tgt}, {which} (sorted by median-guess pMAE, most spread first)")
            print(t.drop(columns=["target", "set"]).round(
                {"mean": 1, "median": 1, "CV": 2, "MG_pMAE%": 1, "IQR/med": 2}).to_string(index=False))
    out = HERE / "results" / "spread.csv"
    out.parent.mkdir(exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\n-> {out}")


# ======================================================================
# sameset
# ======================================================================

def _manifest_units(run: str, dataset: str) -> list:
    mans = sorted((HERE / "results" / run).glob("manifest_*.json"))
    if not mans:
        raise SystemExit(f"no manifest in results/{run}")
    m = json.loads(mans[-1].read_text())
    if dataset not in m["datasets"]:
        raise SystemExit(f"{run}: dataset {dataset!r} not in manifest ({list(m['datasets'])})")
    ids = m["datasets"][dataset]["sample_ids"]
    if dataset != "dfood_tj":
        out = [str(s) for s in ids]
        print(f"{run:28s} n={len(out):4d}  first: {out[:3]}  ({mans[-1].name})")
        return out
    view = m.get("dfood_views") or "?"
    out = []
    for sid in ids:                       # a pair = two consecutive image ids
        nums = [int(x) for x in str(sid).split("#")[0].split("+")]
        meal = min(nums)
        if len(nums) == 1 and view == "oblique":
            meal -= 1                     # oblique = higher id of the pair
        out.append(meal)
    print(f"{run:28s} view={view:9s} n={len(out):4d}  first: {out[:5]}  ({mans[-1].name})")
    return out


def cmd_sameset(a):
    ds = a.dataset or "dfood_tj"
    sets = {r: _manifest_units(r, ds) for r in a.runs}
    ref_run, ref = next(iter(sets.items()))
    ok = True
    for r, s in sets.items():
        if set(s) != set(ref):
            ok = False
            print(f"DIFFERENT: {r} vs {ref_run}: only in {r}: {sorted(set(s) - set(ref))[:10]}, "
                  f"only in {ref_run}: {sorted(set(ref) - set(s))[:10]}")
        if len(set(s)) != len(s):
            ok = False
            print(f"DUPLICATE units in {r}")
    if a.contains:
        old = set(_manifest_units(a.contains, ds))
        missing = old - set(ref)
        print(f"{a.contains} inside {ref_run}: {len(old) - len(missing)}/{len(old)}"
              + (f", missing {sorted(missing)[:10]}" if missing else ""))
    print("SAME set in all runs" if ok else "NOT the same set")


# ======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="paired comparison of runs on the same meals")
    c.add_argument("specs", nargs="+")
    c.add_argument("--target", default="kcal")
    c.add_argument("--dataset")
    c.add_argument("--calibrate", action="store_true",
                   help="remove each spec's overall level first (leave-one-out scale)")
    c.add_argument("--boot", type=int, default=5000)
    c.set_defaults(fn=cmd_compare)

    d = sub.add_parser("decompose", help="split errors into portion (grams) and density")
    d.add_argument("specs", nargs="+")
    d.add_argument("--target", default="kcal", choices=["kcal", "carb_g", "protein_g", "fat_g"])
    d.add_argument("--dataset")
    d.add_argument("--only-ids", metavar="RUN", help="use only the meals of this run")
    d.add_argument("--dfood-root", default="DFood-Tj")
    d.set_defaults(fn=cmd_decompose)

    s = sub.add_parser("spread", help="ground-truth spread per dataset")
    s.add_argument("--targets", nargs="+", default=["kcal", "carb_g"])
    s.add_argument("--dataset")
    s.add_argument("--bench", default="benchmark.py")
    s.set_defaults(fn=cmd_spread)

    m = sub.add_parser("sameset", help="did runs use the same meals?")
    m.add_argument("runs", nargs="+")
    m.add_argument("--dataset", help="default dfood_tj")
    m.add_argument("--contains", metavar="RUN", help="older run whose meals must all be included")
    m.set_defaults(fn=cmd_sameset)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
