from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TARGETS = ["kcal", "mass_g", "carb_g", "protein_g", "fat_g"]
GROUP = ["dataset", "model_label", "prompt_variant", "output_mode"]

try:
    from scipy.stats import t as _student_t
    P_METHOD = "Student t, df = n - 1"

    def p_two_sided(z, n):
        return float(2 * _student_t.sf(abs(z), max(n - 1, 1)))
except Exception:
    P_METHOD = "normal approximation (pip install scipy for Student t)"

    def p_two_sided(z, n):
        return math.erfc(abs(z) / math.sqrt(2))


def read_summaries(paths: list[str]) -> pd.DataFrame:
    frames = []
    for s in paths:
        p = Path(s)
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        if "run" not in df.columns:
            df.insert(0, "run", p.resolve().parent.name)
            df.attrs["added_run"] = True
        df["_source"] = str(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def results_path(run: str, source: str, results_dir: Path) -> Path | None:
    for c in (results_dir / run / "results.jsonl", Path(source).resolve().parent / "results.jsonl"):
        if c.exists():
            return c
    return None


def manifest_view(res: Path) -> str | None:
    mans = sorted(res.parent.glob("manifest_*.json"))
    if not mans:
        return None
    try:
        return json.loads(mans[-1].read_text()).get("dfood_views")
    except Exception:
        return None


def unit_of(dataset: str, sid, view: str | None) -> str:
    s = str(sid)
    if dataset != "dfood_tj":
        return s
    nums = [int(x) for x in re.findall(r"\d+", s.split("#")[0])]
    if not nums:
        return s
    meal = min(nums)
    if len(nums) == 1 and view == "oblique":
        meal -= 1
    return str(meal)


_CACHE: dict = {}


def load_group(res: Path, dataset, model, prompt, output) -> pd.DataFrame | None:
    if res not in _CACHE:
        rows = [json.loads(l) for l in res.open() if l.strip()]
        df = pd.DataFrame(rows)
        for c in ("prompt_variant", "output_mode"):
            df[c] = df[c].fillna("?") if c in df else "?"
        if "repeat" not in df:
            df["repeat"] = 0
        df = df.drop_duplicates(GROUP + ["sample_id", "repeat"], keep="last")
        _CACHE[res] = (df, manifest_view(res))
    df, view = _CACHE[res]
    g = df[(df["dataset"] == dataset) & (df["model_label"] == model)
           & (df["prompt_variant"] == prompt) & (df["output_mode"] == output)]
    if g.empty:
        return None
    lat = g.groupby(g["sample_id"].astype(str))["latency_s"].mean() if "latency_s" in g else None
    recs = {}
    for _, r in g.iterrows():
        sid = str(r["sample_id"])
        d = recs.setdefault(sid, {"n_calls": 0, "n_ok": 0})
        ok = bool(r["parse_ok"])
        d["n_calls"] += 1
        d["n_ok"] += int(ok)
        pred, gt = (r["pred"] or {}) if ok else {}, r["gt"] or {}
        for k in TARGETS:
            tv = gt.get(k)
            if tv is None:
                continue
            pv = pred.get(k)
            if ok and pv is None:
                continue
            d[f"t_{k}"] = float(tv)
            d.setdefault(f"z_{k}", []).append(float(pv) if ok else 0.0)
            if ok:
                d.setdefault(f"p_{k}", []).append(float(pv))
    out = pd.DataFrame({sid: {k: (float(np.mean(v)) if isinstance(v, list) else v)
                              for k, v in d.items()} for sid, d in recs.items()}).T
    for k in TARGETS:
        if f"p_{k}" not in out:
            out = out.drop(columns=[c for c in (f"z_{k}", f"t_{k}") if c in out])
    if lat is not None:
        out["latency_s"] = lat.reindex(out.index)
    out.index.name = "sample_id"
    out["unit"] = [unit_of(dataset, s, view) for s in out.index]
    return out


def ratio_stats(x: np.ndarray, t: np.ndarray):
    n = len(t)
    R = x.sum() / t.sum()
    u = x - R * t
    se = 100 * math.sqrt((u ** 2).sum() / (n * (n - 1))) / t.mean() if n > 1 else float("nan")
    return 100 * R, se, u


def metric_arrays(d: pd.DataFrame, k: str, kind: str):
    sub = d[[f"p_{k}", f"t_{k}"]].dropna() if f"p_{k}" in d and f"t_{k}" in d else None
    if sub is None or len(sub) < 3:
        return None
    p, t = sub[f"p_{k}"].to_numpy(float), sub[f"t_{k}"].to_numpy(float)
    x = np.abs(p - t) if kind == "pMAE" else p - t
    return sub.index, x, t


def metric_arrays_all(d: pd.DataFrame, k: str):
    sub = d[[f"z_{k}", f"t_{k}"]].dropna() if f"z_{k}" in d and f"t_{k}" in d else None
    if sub is None or len(sub) < 3:
        return None
    z, t = sub[f"z_{k}"].to_numpy(float), sub[f"t_{k}"].to_numpy(float)
    return sub.index, np.abs(z - t), t


METRICS = [(k, kind) for k in TARGETS for kind in ("pMAE", "bias")]


def mname(k, kind):
    return f"{k}_pMAE%" if kind == "pMAE" else f"{k}_bias%"


def run_metrics(d: pd.DataFrame) -> dict:
    out = {}
    for k, kind in METRICS:
        a = metric_arrays(d, k, kind)
        if a is None:
            continue
        _, x, t = a
        v, se, _ = ratio_stats(x, t)
        out[mname(k, kind)] = {"value": v, "SE": se, "n": len(t)}
    if "latency_s" in d and d["latency_s"].notna().sum() > 2:
        L = d["latency_s"].dropna().to_numpy(float)
        out["latency_rel%"] = {"value": L.mean(), "SE": 100 * L.std(ddof=1) / math.sqrt(len(L)) / L.mean(),
                               "n": len(L)}
    return out


def paired(dA: pd.DataFrame, dB: pd.DataFrame, k: str, kind: str, min_common: int):
    a, b = metric_arrays(dA, k, kind), metric_arrays(dB, k, kind)
    if a is None or b is None:
        return None
    ia, xa, ta = a
    ib, xb, tb = b
    ua = pd.DataFrame({"x": xa, "t": ta}, index=dA.loc[ia, "unit"].to_numpy())
    ub = pd.DataFrame({"x": xb, "t": tb}, index=dB.loc[ib, "unit"].to_numpy())
    ua = ua[~ua.index.duplicated()]
    ub = ub[~ub.index.duplicated()]
    common = ua.index.intersection(ub.index)
    if len(common) < min_common:
        return None
    ua, ub = ua.loc[common], ub.loc[common]
    same_t = np.isclose(ua["t"], ub["t"], rtol=1e-6, atol=1e-6)
    if not same_t.all():
        ua, ub = ua[same_t], ub[same_t]
        if len(ua) < min_common:
            return None
    t = ua["t"].to_numpy()
    vA, seA, uA = ratio_stats(ua["x"].to_numpy(), t)
    vB, seB, uB = ratio_stats(ub["x"].to_numpy(), t)
    dlt, se_d, _ = ratio_stats(ub["x"].to_numpy() - ua["x"].to_numpy(), t)
    rho = float(np.corrcoef(uA, uB)[0, 1]) if uA.std() > 0 and uB.std() > 0 else float("nan")
    z = dlt / se_d if se_d > 0 else (0.0 if abs(dlt) < 1e-9 else float("nan"))
    return {"n_common": len(t), "A": vA, "B": vB, "delta_pp": dlt, "SE_A": seA, "SE_B": seB,
            "SE_diff_paired": se_d, "SE_diff_unpaired": math.sqrt(seA ** 2 + seB ** 2),
            "rho": rho, "z": z, "p_paired": p_two_sided(z, len(t)) if z == z else float("nan"),
            "dropped_other_truth": int((~same_t).sum())}


def stars(p):
    if p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="summary.csv file(s); merged files need a 'run' column")
    ap.add_argument("--results", default=str(HERE / "results"), help="folder with <run>/results.jsonl")
    ap.add_argument("--deltas", nargs="+", type=float, default=[1, 2, 3, 5, 10, 20, 30, 40])
    ap.add_argument("--min-common", type=int, default=10, help="shared meals needed for a pair")
    ap.add_argument("--min-n", type=int, default=10, help="skip rows with fewer meals (smoke tests)")
    ap.add_argument("--out", default="significance")
    a = ap.parse_args()

    summ = read_summaries(a.summaries)
    results_dir = Path(a.results)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"p values: {P_METHOD}\n")

    data, metr, extra_rows, long_rows = {}, {}, [], []
    for i, row in summ.iterrows():
        res = results_path(row["run"], row["_source"], results_dir)
        key = i
        extra = {}
        if res is None:
            print(f"  {row['run']}: results.jsonl not found -> no SE")
        else:
            d = load_group(res, row["dataset"], row["model"], row["prompt"], row.get("output", "?") or "?")
            if d is None or len(d) < a.min_n:
                print(f"  {row['run']} / {row['dataset']} / {row['model']} / {row['prompt']}: "
                      + ("no matching records" if d is None else f"only {len(d)} meals (< --min-n) -> skipped"))
            else:
                data[key] = d
                metr[key] = run_metrics(d)
                for k in TARGETS:
                    c = f"{k}_pMAE%"
                    if c in row and row[c] not in ("", None) and c in metr[key]:
                        if abs(float(row[c]) - metr[key][c]["value"]) > 0.15:
                            print(f"  WARNING {row['run']} {c}: summary {row[c]} vs recomputed "
                                  f"{metr[key][c]['value']:.2f}")
                for k in TARGETS:
                    m, b = metr[key].get(f"{k}_pMAE%"), metr[key].get(f"{k}_bias%")
                    if m:
                        extra[f"{k}_pMAE_SE"] = round(m["SE"], 2)
                    if b:
                        extra[f"{k}_bias%"] = round(b["value"], 1)
                        extra[f"{k}_bias%_SE"] = round(b["SE"], 2)
        extra_rows.append(extra)

    pair_rows = []
    keys = list(data)
    for ka, kb in combinations(keys, 2):
        ra, rb = summ.loc[ka], summ.loc[kb]
        if ra["dataset"] != rb["dataset"]:
            continue
        for k, kind in METRICS:
            r = paired(data[ka], data[kb], k, kind, a.min_common)
            if r is None:
                continue
            pair_rows.append({"dataset": ra["dataset"], "metric": mname(k, kind),
                              "run_A": ra["run"], "model_A": ra["model"], "prompt_A": ra["prompt"],
                              "run_B": rb["run"], "model_B": rb["model"], "prompt_B": rb["prompt"],
                              **{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()},
                              "sig": stars(r["p_paired"])})
    pairs = pd.DataFrame(pair_rows)
    rho_typ = (pairs.groupby(["dataset", "metric"])["rho"].median().to_dict()
               if len(pairs) else {})

    for key in keys:
        row = summ.loc[key]
        for mn, s in metr[key].items():
            se, n = s["SE"], s["n"]
            rho = rho_typ.get((row["dataset"], mn), float("nan")) if mn != "latency_rel%" else float("nan")
            rec = {"run": row["run"], "dataset": row["dataset"], "model": row["model"],
                   "prompt": row["prompt"], "metric": mn, "n": n,
                   "value": round(s["value"], 2), "SE": round(se, 2),
                   "rho_typical": round(rho, 3) if rho == rho else "",
                   "MDD95_unpaired": round(1.96 * math.sqrt(2) * se, 1)}
            se_p = se * math.sqrt(max(2 * (1 - rho), 0)) if rho == rho else float("nan")
            rec["MDD95_paired"] = round(1.96 * se_p, 1) if se_p == se_p else ""
            for dl in a.deltas:
                lab = f"{dl:g}"
                rec[f"p_unpaired_{lab}pp"] = float(f"{p_two_sided(dl / (se * math.sqrt(2)), n):.3g}") if se > 0 else ""
                rec[f"p_paired_{lab}pp"] = (float(f"{p_two_sided(dl / se_p, n):.3g}")
                                           if se_p == se_p and se_p > 0 else "")
            long_rows.append(rec)
            if mn.endswith("_pMAE%"):
                base = mn[:-len("_pMAE%")]
                ex = extra_rows[list(summ.index).index(key)]
                ex[f"{base}_pMAE_MDD_unpaired"] = rec["MDD95_unpaired"]
                ex[f"{base}_pMAE_MDD_paired"] = rec["MDD95_paired"]

    orig_cols = [c for c in summ.columns if c != "_source"]
    wide = summ[orig_cols].copy()
    ex = pd.DataFrame(extra_rows, index=summ.index)
    order = []
    for k in TARGETS:
        order += [c for c in (f"{k}_pMAE_SE", f"{k}_pMAE_MDD_unpaired", f"{k}_pMAE_MDD_paired",
                              f"{k}_bias%", f"{k}_bias%_SE") if c in ex]
    wide = pd.concat([wide, ex[order]], axis=1)
    wide.to_csv(out_dir / "summary_with_se.csv", index=False)
    long = pd.DataFrame(long_rows)
    long.to_csv(out_dir / "p_by_delta.csv", index=False)
    pairs.to_csv(out_dir / "pairs.csv", index=False)

    if len(long):
        print("p value of a difference of Δ pp (paired = same meals, typical rho of the dataset)")
        print("  * p<0.05  ** p<0.01  *** p<0.001\n")
        for (run, ds, model, prompt), g in long.groupby(["run", "dataset", "model", "prompt"], sort=False):
            print(f"{run} | {ds} | {model} | {prompt}   n={g['n'].iloc[0]}")
            hdr = "".join(f"{f'{d:g}pp':>9s}" for d in a.deltas)
            print(f"  {'metric':16s}{'value':>8s}{'SE':>6s} {'rho':>5s}  {'':8s}{hdr}")
            for _, r in g.iterrows():
                if not str(r["metric"]).endswith("pMAE%"):
                    continue
                for mode in ("unpaired", "paired"):
                    ps = [r[f"p_{mode}_{d:g}pp"] for d in a.deltas]
                    if all(p == "" for p in ps):
                        continue
                    cells = "".join(f"{(f'{p:.3f}' + stars(p)) if p != '' else '':>9s}" for p in ps)
                    lead = (f"  {r['metric']:16s}{r['value']:8.1f}{r['SE']:6.1f} "
                            f"{str(r['rho_typical'])[:5]:>5s}" if mode == "unpaired"
                            else f"  {'':16s}{'':8s}{'':6s} {'':5s}")
                    print(f"{lead}  {mode:8s}{cells}")
            print()
        pm = long[long["metric"].str.endswith("_pMAE%")].copy()
        pm["k"] = pm["SE"] * np.sqrt(pm["n"]) / pm["value"]
        print("shortcut SE(pMAE) ~= k * pMAE / sqrt(n):  median k per dataset")
        for ds, g in pm.groupby("dataset"):
            print(f"  {ds:20s} k = {g['k'].median():.2f}  (range {g['k'].min():.2f}..{g['k'].max():.2f})")
    if len(pairs):
        print(f"\nreal pairs ({len(pairs)} rows, pMAE only shown; all in pairs.csv):")
        for _, r in pairs[pairs["metric"].str.endswith("pMAE%")].iterrows():
            print(f"  {r['dataset']:12s} {r['metric']:14s} {r['run_A'][:34]:34s} -> {r['run_B'][:34]:34s} "
                  f"n={r['n_common']:3d}  Δ {r['delta_pp']:+6.1f} pp  SE {r['SE_diff_paired']:4.1f}  "
                  f"rho {r['rho']:+.2f}  p {r['p_paired']:.3g} {r['sig']}")
    print(f"\n-> {out_dir}/summary_with_se.csv, p_by_delta.csv, pairs.csv")


if __name__ == "__main__":
    main()
