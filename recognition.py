#!/usr/bin/env python3
"""
recognition.py - food / ingredient recognition of the decision measurement (no new
estimation runs): the item names the models gave in the final P3_macros runs are compared
with the reference names of the meals.

    venv/bin/python recognition.py dict --n5k-root PATH --dfood-root PATH \
                                        --snapme-root PATH --acetada-root PATH
    venv/bin/python recognition.py map --mapper flash38      # API calls
    venv/bin/python recognition.py score --mapper flash38    # no API calls
    venv/bin/python recognition.py agree                     # the two mappings compared
  (--mapper flash-lite, the default, is the first mapping: mapping.jsonl, scores.csv)

Protocol (fixed before the mapping run, 2026-11-23; commit aec291a). Extended on
2026-11-23, after the Nutrition5k / DFood-Tj scoring (commit b6452e6), with the same
protocol to the two other image datasets that have item-level reference names (SNAPMe,
ACETADA - the second level of the 2026-11-21 plan) and with the manual check of the
mapping (commit fad60f3). OmniFood8k has no item names and is not evaluated.
Changed on 2026-11-23 evening, after the mapping of all four datasets (commit 4839e0f):
a spot check of the Flash-Lite mappings showed many errors, so the mapping is also done
by a stronger model, Gemini 3.8 Flash (the measurement's setting: thinking level low), with
the same prompt, dictionary, batches and threshold; its result is the main result, the
Flash-Lite mapping is reported for comparison. The manual check is dropped (the
2026-11-18 decision: no manual annotation; the two mappings are compared instead).
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
  - Mapping: a language model with the measurement's settings and JSON schema, called
    through benchmark.call_with_retries: Gemini 3.8 Flash (thinking level low; main) and
    Gemini 3.5 Flash-Lite (temperature 0.1; comparison). The model sees only the item
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
  - Agreement of the two mappings (agree): per dataset, the share of the distinct item
    names, and of the item occurrences in the scored runs, for which the two mappings
    give the same set of dictionary entries (after the 85 threshold).

Output (folder recognition/):
  reference.csv     dataset, sample_id, name (cleaned) for the tested meals
  dictionary_<dataset>.txt
  items.csv         run, dataset, sample_id, item (raw), name (cleaned)
  mapping[_flash38].jsonl   one record per API call (appended; finished batches skipped)
  manifest[_flash38].json   the mapping settings (model entry, prompt template, batch
                            size, dictionary sha256), one entry per invocation
  scores[_flash38].csv      run x threshold: recall and precision with 95% CI, counts
  datasets[_flash38].csv    per dataset: dictionary size, distinct item names, names with
                            no entry, returned entries, entries equal to a dictionary entry
  agreement.csv             per dataset: agreement of the two mappings
  agreement_types.csv       per dataset: how the two mappings of the distinct names differ
                            (same set, both empty, only one mapping has entries, one set
                            contains the other, other), the share of the Flash-Lite-only
                            names with >= 2 entries, and the mean number of entries per name
  (no suffix: the Flash-Lite mapping; _flash38: the Gemini 3.8 Flash mapping)
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

BATCH = 25
MAIN_THRESHOLD = 85
THRESHOLDS = (80, 85, 90, 95)
MAPPERS = {
    "flash-lite": dict(label="gemini-flash-lite", provider="gemini",
                       model="gemini-3.5-flash-lite", temperature=0.1, extra={},
                       output_mode="schema", min_interval_s=6.0),
    "flash38": dict(label="gemini-3.8-flash-low", provider="gemini",
                    model="gemini-3.8-flash", temperature=None,
                    extra={"thinking_config": {"thinking_level": "low"}},
                    output_mode="schema", min_interval_s=12.0)}
MAIN_MAPPER = "flash38"


def suffix(mapper: str) -> str:
    return "" if mapper == "flash-lite" else f"_{mapper}"

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
    model, sfx = MAPPERS[a.mapper], suffix(a.mapper)
    manifest = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model,
                "batch": BATCH, "prompt": PROMPT, "schema": SCHEMA,
                "benchmark_sha256": hashlib.sha256(
                    (HERE / "benchmark.py").read_bytes()).hexdigest(),
                "dictionary_sha256": {}}
    done = set()
    log = OUT / f"mapping{sfx}.jsonl"
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
            r = B.call_with_retries(model, prompt, [], SCHEMA, last_call)
            r.pop("traceback", None)
            rec = {"dataset": ds, "batch": i, "names_sha256": h, "names": names,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **r}
            with log.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{ds} {i + 1}/{len(bs)} {'ok' if r['api_ok'] else r['error']}")
    mpath = OUT / f"manifest{sfx}.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else None
    runs = (old["invocations"] if old and "invocations" in old else [old] if old else [])
    mpath.write_text(json.dumps({"invocations": runs + [manifest]}, indent=2,
                                ensure_ascii=False))


# ---------------------------------------------------------------- score
def mapped_entries(mapper: str = "flash-lite") -> tuple[dict, dict]:
    """(dataset, name) -> list of returned entries; plus counts of the parsing."""
    out, stats = {}, {"names": 0, "by_text": 0, "by_position": 0, "missing": 0}
    for line in (OUT / f"mapping{suffix(mapper)}.jsonl").open():
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
    sfx = suffix(a.mapper)
    entries, pstats = mapped_entries(a.mapper)
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
    write_csv(OUT / f"datasets{sfx}.csv", list(dstat.values()))
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
    write_csv(OUT / f"scores{sfx}.csv", rows)
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


def mapped_after_threshold(mapper: str) -> dict:
    """(dataset, name) -> set of dictionary entries kept with the main threshold."""
    from rapidfuzz import fuzz, process
    words = {ds: (OUT / f"dictionary_{ds}.txt").read_text().split("\n")[:-1] for ds in RUNS}
    entries, _ = mapped_entries(mapper)
    out = {}
    for (ds, name), es in entries.items():
        kept = set()
        for e in es:
            m = process.extractOne(clean(e), words[ds], scorer=fuzz.token_sort_ratio)
            if m[1] >= MAIN_THRESHOLD:
                kept.add(m[0])
        out[(ds, name)] = kept
    return out


def cmd_agree(a) -> None:
    main, other = mapped_after_threshold(MAIN_MAPPER), mapped_after_threshold("flash-lite")
    items = read_csv(OUT / "items.csv")
    rows = []
    for ds in RUNS:
        names = sorted(n for d, n in main if d == ds)
        same = [main[(ds, n)] == other[(ds, n)] for n in names]
        occ = [main[(ds, r["name"])] == other[(ds, r["name"])]
               for r in items if r["dataset"] == ds]
        rows.append({"dataset": ds, "names": len(names),
                     "names_same_%": round(100 * sum(same) / len(same), 1),
                     "items": len(occ), "items_same_%": round(100 * sum(occ) / len(occ), 1)})
        print("  ".join(f"{k} {v}" for k, v in rows[-1].items()))
    write_csv(OUT / "agreement.csv", rows)
    types = []
    for ds in RUNS:
        names = sorted(n for d, n in main if d == ds)
        count = dict.fromkeys(["same", "same_empty", "flash_lite_only", "flash38_only", "flash_lite_more",
                               "flash38_more", "other"], 0)
        lite_only_2plus = 0
        for n in names:
            m, o = main[(ds, n)], other[(ds, n)]
            if m == o:
                k = "same" if m else "same_empty"
            elif o and not m:
                k = "flash_lite_only"
                lite_only_2plus += len(o) >= 2
            elif m and not o:
                k = "flash38_only"
            elif m < o:
                k = "flash_lite_more"
            elif o < m:
                k = "flash38_more"
            else:
                k = "other"
            count[k] += 1
        row = {"dataset": ds, "names": len(names)}
        row.update({f"{k}_%": round(100 * v / len(names), 1) for k, v in count.items()})
        row["flash_lite_only_2plus_%"] = round(100 * lite_only_2plus / max(count["flash_lite_only"], 1), 1)
        row["entries_per_name_flash_lite"] = round(sum(len(other[(ds, n)]) for n in names) / len(names), 2)
        row["entries_per_name_flash38"] = round(sum(len(main[(ds, n)]) for n in names) / len(names), 2)
        types.append(row)
        print("  ".join(f"{k} {v}" for k, v in row.items()))
    write_csv(OUT / "agreement_types.csv", types)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dict")
    for name in ("n5k", "dfood", "snapme", "acetada"):
        p.add_argument(f"--{name}-root", required=True)
    for name in ("map", "score"):
        sub.add_parser(name).add_argument("--mapper", choices=list(MAPPERS),
                                          default="flash-lite")
    sub.add_parser("agree")
    a = ap.parse_args()
    {"dict": cmd_dict, "map": cmd_map, "score": cmd_score, "agree": cmd_agree}[a.cmd](a)


if __name__ == "__main__":
    main()
