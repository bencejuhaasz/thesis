#!/usr/bin/env python3
"""
recognition.py - food / ingredient recognition of the decision measurement (no new
estimation runs): the item names the models gave in the final P3_macros runs are compared
with the reference names of the meals.

    venv/bin/python recognition.py dict --n5k-root PATH --dfood-root PATH \
                                        --snapme-root PATH --acetada-root PATH
    venv/bin/python recognition.py map            # API calls (Gemini Flash-Lite)
    venv/bin/python recognition.py score          # no API calls
    venv/bin/python recognition.py sample         # manual check of the mapping: sample
    venv/bin/python recognition.py validate       # manual check of the mapping: rates

Protocol (fixed before the mapping run, 2026-11-23; commit aec291a). Extended on
2026-11-23, after the Nutrition5k / DFood-Tj scoring (commit b6452e6), with the same
protocol to the two other image datasets that have item-level reference names (SNAPMe,
ACETADA - the second level of the 2026-11-21 plan) and with the manual check of the
mapping (sample / validate). OmniFood8k has no item names and is not evaluated.
  - Runs: final_{n5k,dfood,acetada}_{lite,sonnet,sol}_macros and final_snapme_lite_macros
    (final_snapme_lite_hint is left out: its input contains the reference names), the
    tested positions only (the same meals as the descriptive results of the decision
    measurement: prereg_tests.load_run). Item names: the parsed answer's item_names
    (pred.item_names) of the meal's record.
  - Dictionary: every reference name of the dataset (Nutrition5k: ingr_name; DFood-Tj:
    food_names split at '; '; SNAPMe: Food_Description of the items with a photo;
    ACETADA: the food item names, as given - compound names are not split), cleaned:
    Unicode NFKC, lowercase, whitespace collapsed, stripped, duplicates removed. The
    meal's reference = its cleaned names (a set).
  - Mapping: Gemini 3.5 Flash-Lite with the measurement's settings (temperature 0.1, JSON
    schema), called through benchmark.call_with_retries. The model sees only the item
    names (25 per call, each distinct cleaned name once per dataset, sorted) and the
    whole dictionary - not the image and not the meal's reference. PROMPT below.
  - Matching: every entry the model returned is matched to the closest dictionary entry
    with rapidfuzz fuzz.token_sort_ratio (word order ignored, a subset is not a full
    match); kept if the score is >= the threshold. Main threshold: 85. Sensitivity:
    80, 85, 90, 95. Reported: the share of entries that equal a dictionary entry after
    cleaning.
  - Metrics per run (ratios of sums over meals, 95% CI with the delta method and Student
    t, as for pMAE: significance.ratio_stats):
      recall    = reference names covered by the mapped entries of the meal's items
                  / all reference names
      precision = items with at least one mapped entry in the meal's reference
                  / all items (an item with no mapped entry counts as wrong)
    Descriptive only, no hypothesis test.
  - Manual check of the mapping (sample / validate): per dataset 50 item occurrences
    drawn uniformly from the scored runs' items (random.Random(42), sorted input). For
    each, the person checks the mapped dictionary entries (after the 85 threshold)
    against the whole dictionary with the rule of the prompt (an entry fits if the item
    name names it or unambiguously denotes it) and writes one verdict: ok (exactly the
    fitting entries), missing (a fitting entry is left out, nothing wrong added), wrong
    (an entry that does not fit is added, nothing missing), both. Reported per dataset:
    the share of each verdict.

Output (folder recognition/):
  reference.csv     dataset, sample_id, name (cleaned) for the tested meals
  dictionary_<dataset>.txt
  items.csv         run, dataset, sample_id, item (raw), name (cleaned)
  mapping.jsonl     one record per API call (appended; finished batches are skipped)
  manifest.json     the mapping settings (model entry, prompt template, batch size,
                    dictionary sha256)
  scores.csv        run x threshold: recall and precision with 95% CI, counts
  datasets.csv      per dataset: dictionary size, distinct item names, names with no
                    entry, returned entries, entries equal to a dictionary entry
  validation.csv    the manual-check sample (verdict column filled in by hand)
  validation_rates.csv  per dataset: share of each verdict
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "recognition"
RESULTS = HERE / "results"
RUNS = {"nutrition5k": ["final_n5k_lite_macros", "final_n5k_sonnet_macros",
                        "final_n5k_sol_macros"],
        "dfood_tj": ["final_dfood_lite_macros", "final_dfood_sonnet_macros",
                     "final_dfood_sol_macros"],
        "snapme": ["final_snapme_lite_macros"],
        "acetada": ["final_acetada_lite_macros", "final_acetada_sonnet_macros",
                    "final_acetada_sol_macros"]}
SAMPLE = 50
VERDICTS = ("ok", "missing", "wrong", "both")
BATCH = 25
MAIN_THRESHOLD = 85
THRESHOLDS = (80, 85, 90, 95)
MODEL = dict(label="gemini-flash-lite", provider="gemini", model="gemini-3.5-flash-lite",
             temperature=0.1, extra={}, output_mode="schema", min_interval_s=6.0)

PROMPT = """You map food item names to a fixed dictionary.

Dictionary (one entry per line):
{dictionary}

For each item name below, list every dictionary entry that the item names or unambiguously denotes. An item may correspond to several entries (for example a mixed dish) or to none. Use the entries exactly as written; do not invent new entries and do not add ingredients that the item name does not mention.

Items:
{items}

Return JSON: {{"mappings": [{{"item": "...", "entries": ["..."]}}]}}"""

SCHEMA = {"type": "object",
          "properties": {"mappings": {"type": "array", "items": {
              "type": "object",
              "properties": {"item": {"type": "string"},
                             "entries": {"type": "array", "items": {"type": "string"}}},
              "required": ["item", "entries"]}}},
          "required": ["mappings"]}


def clean(s) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).strip().lower()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def tested_items() -> list[dict]:
    """run, dataset, sample_id, item for every parsed tested meal of the six runs."""
    sys.path.insert(0, str(HERE))
    import prereg_tests as P
    rows = []
    for ds, runs in RUNS.items():
        for run in runs:
            d, ds_run, _ = P.load_run(run, RESULTS)
            assert ds_run == ds, (run, ds_run)
            tested = set(map(str, d.index[d["n_ok"] > 0]))
            last = {}
            for line in (RESULTS / run / "results.jsonl").open():
                r = json.loads(line)
                if r.get("pred") and str(r["sample_id"]) in tested:
                    last[str(r["sample_id"])] = r     # the last parsed record counts
            if set(last) != tested:
                raise SystemExit(f"{run}: {len(tested - set(last))} tested meals without items")
            for sid in sorted(last):
                for item in last[sid]["pred"].get("item_names") or []:
                    if clean(item):
                        rows.append({"run": run, "dataset": ds, "sample_id": sid,
                                     "item": item, "name": clean(item)})
    return rows


# ---------------------------------------------------------------- dict
def loader(folder: str):
    """The dataset_loaders/<folder>/loader.py module (each folder has its own loader.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"loader_{folder}", HERE / "dataset_loaders" / folder / "loader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_dict(a) -> None:
    _, ingr = loader("nutrition5k").load_nutrition5k(a.n5k_root)
    dfood = loader("dfood-tj")
    images, _ = dfood.load_fnutr(a.dfood_root)
    _id_key = dfood._id_key
    snap_items, _ = loader("snapme").load_snapme(a.snapme_root)
    ace_meals, ace_items = loader("acetada").load_acetada(a.acetada_root)

    ref = {ds: {} for ds in RUNS}
    for sid, name in zip(ingr["dish_id"], ingr["ingr_name"]):
        if clean(name):
            ref["nutrition5k"].setdefault(str(sid), set()).add(clean(name))
    for iid, names in zip(images["image_id"], images["food_names"]):
        if isinstance(names, str):
            ref["dfood_tj"][_id_key(iid)] = {clean(n) for n in names.split("; ") if clean(n)}
    for fn, name in zip(snap_items["filename"], snap_items["Food_Description"]):
        if isinstance(fn, str) and clean(name):
            ref["snapme"].setdefault(fn, set()).add(clean(name))
    for mi, name in zip(ace_items["meal_idx"], ace_items["name"]):
        if isinstance(name, str) and clean(name):
            sid = str(ace_meals.loc[mi, "before_filepath"])
            ref["acetada"].setdefault(sid, set()).add(clean(name))

    OUT.mkdir(exist_ok=True)
    items = tested_items()
    write_csv(OUT / "items.csv", items)
    rows = []
    for ds in RUNS:
        words = sorted(set().union(*ref[ds].values()))
        (OUT / f"dictionary_{ds}.txt").write_text("\n".join(words) + "\n")
        sids = sorted({r["sample_id"] for r in items if r["dataset"] == ds})
        missing = [s for s in sids if s not in ref[ds]]
        if missing:
            raise SystemExit(f"{ds}: no reference names for {len(missing)} meals, "
                             f"e.g. {missing[:5]}")
        rows += [{"dataset": ds, "sample_id": s, "name": n}
                 for s in sids for n in sorted(ref[ds][s])]
        print(f"{ds}: dictionary {len(words)} names; {len(sids)} tested meals, "
              f"{sum(len(ref[ds][s]) for s in sids)} reference names; "
              f"{sum(r['dataset'] == ds for r in items)} items, "
              f"{len({r['name'] for r in items if r['dataset'] == ds})} distinct")
    write_csv(OUT / "reference.csv", rows)


# ---------------------------------------------------------------- map
def batches(ds: str) -> list[list[str]]:
    names = sorted({r["name"] for r in read_csv(OUT / "items.csv") if r["dataset"] == ds})
    return [names[i:i + BATCH] for i in range(0, len(names), BATCH)]


def cmd_map(a) -> None:
    sys.path.insert(0, str(HERE))
    import benchmark as B
    manifest = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": MODEL,
                "batch": BATCH, "prompt": PROMPT, "schema": SCHEMA,
                "benchmark_sha256": hashlib.sha256(
                    (HERE / "benchmark.py").read_bytes()).hexdigest(),
                "dictionary_sha256": {}}
    done = set()
    log = OUT / "mapping.jsonl"
    if log.exists():
        for line in log.open():
            r = json.loads(line)
            if r["api_ok"]:
                done.add((r["dataset"], r["batch"], r["names_sha256"]))
    last_call: dict = {}
    for ds in RUNS:
        dictionary = (OUT / f"dictionary_{ds}.txt").read_text()
        manifest["dictionary_sha256"][ds] = hashlib.sha256(dictionary.encode()).hexdigest()
        bs = batches(ds)
        for i, names in enumerate(bs):
            h = hashlib.sha256("\n".join(names).encode()).hexdigest()
            if (ds, i, h) in done:
                continue
            prompt = PROMPT.format(dictionary=dictionary.strip(),
                                   items="\n".join(f"{k}. {n}" for k, n in
                                                   enumerate(names, 1)))
            r = B.call_with_retries(MODEL, prompt, [], SCHEMA, last_call)
            r.pop("traceback", None)
            rec = {"dataset": ds, "batch": i, "names_sha256": h, "names": names,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **r}
            with log.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{ds} {i + 1}/{len(bs)} {'ok' if r['api_ok'] else r['error']}")
    old = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else None
    runs = (old["invocations"] if old and "invocations" in old else [old] if old else [])
    (OUT / "manifest.json").write_text(json.dumps({"invocations": runs + [manifest]},
                                                  indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- score
def mapped_entries() -> tuple[dict, dict]:
    """(dataset, name) -> list of returned entries; plus counts of the parsing."""
    out, stats = {}, {"names": 0, "by_text": 0, "by_position": 0, "missing": 0}
    for line in (OUT / "mapping.jsonl").open():
        r = json.loads(line)
        if not r["api_ok"]:
            continue
        resp = json.loads(r["text"])["mappings"]
        by_text = {clean(m["item"]): m["entries"] for m in resp}
        for k, name in enumerate(r["names"]):
            stats["names"] += 1
            if name in by_text:
                out[(r["dataset"], name)] = by_text[name]
                stats["by_text"] += 1
            elif len(resp) == len(r["names"]):
                out[(r["dataset"], name)] = resp[k]["entries"]
                stats["by_position"] += 1
            else:
                out[(r["dataset"], name)] = []
                stats["missing"] += 1
    return out, stats


def ci(x, t) -> tuple[float, float, float]:
    import numpy as np
    from scipy.stats import t as student
    sys.path.insert(0, str(HERE))
    import significance as sig
    v, se, _ = sig.ratio_stats(np.asarray(x, float), np.asarray(t, float))
    q = student.ppf(0.975, len(t) - 1)
    return v, v - q * se, v + q * se


def cmd_score(a) -> None:
    from rapidfuzz import fuzz, process
    items = read_csv(OUT / "items.csv")
    ref: dict = {}
    for r in read_csv(OUT / "reference.csv"):
        ref.setdefault((r["dataset"], r["sample_id"]), set()).add(r["name"])
    words = {ds: (OUT / f"dictionary_{ds}.txt").read_text().split("\n")[:-1] for ds in RUNS}
    entries, pstats = mapped_entries()
    missing = {(r["dataset"], r["name"]) for r in items} - set(entries)
    if missing:
        raise SystemExit(f"{len(missing)} item names are not mapped yet: run 'map'")

    best = {}                       # (dataset, entry) -> (dictionary entry, score)
    n_entries = n_exact = 0
    dstat = {ds: {"dataset": ds, "dictionary": len(words[ds]), "names": 0,
                  "names_no_entry": 0, "entries": 0, "entries_exact": 0} for ds in RUNS}
    for (ds, _), es in entries.items():
        dstat[ds]["names"] += 1
        dstat[ds]["names_no_entry"] += not es
        for e in es:
            n_entries += 1
            n_exact += clean(e) in words[ds]
            dstat[ds]["entries"] += 1
            dstat[ds]["entries_exact"] += clean(e) in words[ds]
            if (ds, clean(e)) not in best:
                m = process.extractOne(clean(e), words[ds], scorer=fuzz.token_sort_ratio)
                best[(ds, clean(e))] = (m[0], m[1])
    write_csv(OUT / "datasets.csv", list(dstat.values()))
    for d in dstat.values():
        print(f"{d['dataset']:12} dictionary {d['dictionary']:5}  item names {d['names']:5}  "
              f"no entry {d['names_no_entry']:5} ({100 * d['names_no_entry'] / d['names']:.1f}%)"
              f"  entries {d['entries']:5}, exact {d['entries_exact']}")
    print(f"item names: {pstats['names']} (matched to the answer by text "
          f"{pstats['by_text']}, by position {pstats['by_position']}, missing "
          f"{pstats['missing']}); returned entries: {n_entries}, equal to a dictionary "
          f"entry after cleaning: {n_exact} ({100 * n_exact / n_entries:.1f}%)")

    rows = []
    for thr in THRESHOLDS:
        for ds, runs in RUNS.items():
            for run in runs:
                meals: dict = {}
                for r in (r for r in items if r["run"] == run):
                    got = {best[(ds, clean(e))][0] for e in entries[(ds, r["name"])]
                           if best[(ds, clean(e))][1] >= thr}
                    m = meals.setdefault(r["sample_id"], {"items": 0, "correct": 0,
                                                          "covered": set()})
                    m["items"] += 1
                    m["correct"] += bool(got & ref[(ds, r["sample_id"])])
                    m["covered"] |= got & ref[(ds, r["sample_id"])]
                sids = sorted(meals)
                rec = ci([len(meals[s]["covered"]) for s in sids],
                         [len(ref[(ds, s)]) for s in sids])
                prec = ci([meals[s]["correct"] for s in sids],
                          [meals[s]["items"] for s in sids])
                rows.append({"threshold": thr, "run": run, "dataset": ds,
                             "meals": len(sids),
                             "reference_names": sum(len(ref[(ds, s)]) for s in sids),
                             "items": sum(meals[s]["items"] for s in sids),
                             "recall_%": round(rec[0], 1), "recall_lo": round(rec[1], 1),
                             "recall_hi": round(rec[2], 1),
                             "precision_%": round(prec[0], 1),
                             "precision_lo": round(prec[1], 1),
                             "precision_hi": round(prec[2], 1)})
    write_csv(OUT / "scores.csv", rows)
    for r in rows:
        if r["threshold"] == MAIN_THRESHOLD:
            print(f"{r['run']:28} meals {r['meals']:3}  recall {r['recall_%']:5.1f} "
                  f"({r['recall_lo']}; {r['recall_hi']})  precision {r['precision_%']:5.1f} "
                  f"({r['precision_lo']}; {r['precision_hi']})")
    for run in [r for rs in RUNS.values() for r in rs]:
        vals = [(r["recall_%"], r["precision_%"]) for r in rows if r["run"] == run]
        print(f"{run:28} thresholds {THRESHOLDS}: recall "
              f"{min(v[0] for v in vals)}-{max(v[0] for v in vals)}, precision "
              f"{min(v[1] for v in vals)}-{max(v[1] for v in vals)}")


def mapped_after_threshold() -> dict:
    """(dataset, name) -> sorted dictionary entries kept with the main threshold."""
    from rapidfuzz import fuzz, process
    words = {ds: (OUT / f"dictionary_{ds}.txt").read_text().split("\n")[:-1] for ds in RUNS}
    entries, _ = mapped_entries()
    out = {}
    for (ds, name), es in entries.items():
        kept = set()
        for e in es:
            m = process.extractOne(clean(e), words[ds], scorer=fuzz.token_sort_ratio)
            if m[1] >= MAIN_THRESHOLD:
                kept.add(m[0])
        out[(ds, name)] = sorted(kept)
    return out


def cmd_sample(a) -> None:
    import random
    path = OUT / "validation.csv"
    if path.exists():
        raise SystemExit(f"{path} exists (it may hold verdicts): not overwritten")
    items = read_csv(OUT / "items.csv")
    mapped = mapped_after_threshold()
    rng = random.Random(42)
    rows = []
    for ds in RUNS:
        pool = sorted((r["run"], r["sample_id"], r["item"], r["name"])
                      for r in items if r["dataset"] == ds)
        for run, sid, item, name in rng.sample(pool, SAMPLE):
            rows.append({"dataset": ds, "run": run, "sample_id": sid, "item": item,
                         "entries": " | ".join(mapped[(ds, name)]), "verdict": "",
                         "note": ""})
    write_csv(path, rows)
    print(f"{len(rows)} rows -> {path}; dictionaries: {OUT}/dictionary_<dataset>.txt; "
          f"verdicts: {', '.join(VERDICTS)}")


def cmd_validate(a) -> None:
    rows = read_csv(OUT / "validation.csv")
    bad = [r for r in rows if r["verdict"].strip() not in VERDICTS]
    if bad:
        raise SystemExit(f"{len(bad)} rows without a valid verdict ({', '.join(VERDICTS)}), "
                         f"e.g. {bad[0]['dataset']} / {bad[0]['item']}")
    out = []
    for ds in RUNS:
        v = [r["verdict"].strip() for r in rows if r["dataset"] == ds]
        rec = {"dataset": ds, "n": len(v)}
        rec.update({f"{k}_%": round(100 * v.count(k) / len(v), 1) for k in VERDICTS})
        out.append(rec)
        print("  ".join(f"{k} {x}" for k, x in rec.items()))
    write_csv(OUT / "validation_rates.csv", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dict")
    for name in ("n5k", "dfood", "snapme", "acetada"):
        p.add_argument(f"--{name}-root", required=True)
    for name in ("map", "score", "sample", "validate"):
        sub.add_parser(name)
    a = ap.parse_args()
    {"dict": cmd_dict, "map": cmd_map, "score": cmd_score, "sample": cmd_sample,
     "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    main()
