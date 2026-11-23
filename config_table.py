#!/usr/bin/env python3
"""
config_table.py - the measurement configuration of every run whose result is reported in
the thesis (F5: reproducible record of the configurations). No API calls.

    venv/bin/python config_table.py

The settings are taken from what the run actually did, not from the run name:
  - dataset, model (API id), prompt variant, input modality, number of images and the
    DFood-Tj view(s): from the call records in results.jsonl ('final' records);
  - temperature and the provider-specific settings (reasoning / effort / thinking level,
    output token limit): from the manifest entry of the model label used in the run;
  - DFood-Tj view of the one-image runs: from the manifest ('dfood_views'), because the
    record stores the view names only for multi-image calls;
  - date: first and last call timestamp;
  - number of meals: distinct sample ids among the final records.
The JPEG quality is not recorded in the manifest or the records; it is a constant of the
program that ran (JPEG_QUALITY), so it is not in the table. For the runs whose program
version is committed here, --versions reports it.

Output (folder config_table/):
  config_table.csv   one row per run
  config_table.tex   LaTeX longtable rows, grouped by where the result is reported

    venv/bin/python config_table.py --versions

additionally writes program_versions.csv: for every listed run, whether the program version
that ran it (the manifest's benchmark_sha256) is one of the versions of the benchmark
programs committed in this git repository, and if so which program and its JPEG_QUALITY.
It needs the git history (it runs git log / git show), so its result is only meaningful
in the repository where the runs were committed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (group title, [(run folder, dataset filter or None), ...]); a run is listed once, in the
# first group where its result is reported.
DS5 = ("nutrition5k", "omnifood8k", "snapme", "acetada", "dfood")
GROUPS: list[tuple[str, list[tuple[str, str | None]]]] = [
    ("Előmérés: modellek (\\ref{tab:elomeres_modellek}.~táblázat)", [
        ("nutrition5k_macros_n50_s42", None),
        ("nutrition5k_flash38_macros_n50_s42", None),
        ("nutrition5k_gpt-luna_macros_n50_s42", None),
        ("nutrition5k_sol_macros_n50_s42", None),
        ("nutrition5k_claude-haiku-low_macros_n50_s42", None),
        ("nutrition5k_sonnet_macros_n50_s42", None),
        ("nutrition5k_claude-opus-low_macros_n50_s42", None),
        ("omnifood8k_allimage_n50_s42", None),
        ("omnifood8k_1image_flash38_n50_s42", None),
        ("omnifood8k_1image_gpt-luna_macros_n50_s42", None),
        ("omnifood8k_1image_sol_n50_s42", None),
        ("omnifood8k_1image_astra_n50_s42", None),
        ("omnifood8k_1image_claude-haiku-low_macros_n50_s42", None),
        ("omnifood8k_1image_sonnet_n50_s42", None),
        ("omnifood8k_1image_claude-opus-low_macros_n50_s42", None),
        ("snapme_lite_macros_n50_s42", None),
        ("snapme_flash38_macros_n50_s42", None),
        ("snapme_gpt-luna_macros_n50_s42", None),
        ("snapme_sol_macros_n50_s42", None),
        ("snapme_claude-haiku-low_macros_n50_s42", None),
        ("snapme_sonnet_macros_n50_s42", None),
        ("snapme_claude-opus-low_macros_n50_s42", None),
        ("acetada_lite_macros_n50_s42", None),
        ("acetada_flash38_macros_n50_s42", None),
        ("acetada_gpt-luna_macros_n50_s42", None),
        ("acetada_gpt-sol_macros_n50_s42", None),
        ("acetada_claude-haiku-low_macros_n50_s42", None),
        ("acetada_sonnet_macros_n50_s42", None),
        ("acetada_claude-opus-low_macros_n50_s42", None),
        ("dfood_lite_macros_oblique_n50_s42", None),
        ("dfood_flash38_macros_oblique_n50_s42", None),
        ("dfood_gpt-luna_macros_oblique_n50_s42", None),
        ("dfood_gpt-sol_macros_oblique_n50_s42", None),
        ("dfood_claude-haiku-low_macros_oblique_n50_s42", None),
        ("dfood_sonnet_macros_oblique_n50_s42", None),
        ("dfood_claude-opus-low_macros_oblique_n50_s42", None),
    ]),
    ("Előmérés: promptváltozatok (\\ref{tab:elomeres_promptok}.~táblázat)", [
        (f"{p}_lite_{v}{'_oblique' if p == 'dfood' else ''}_n50_s42", None)
        for p in DS5 for v in ("recipe", "anchor", "portion", "check", "recipecheck")
    ]),
    ("Előmérés: bemenet és feldolgozás (\\ref{tab:elomeres_eljarasok}.~táblázat)", [
        ("omnifood8k_1image_n50_s42", None),
        ("macros50_35", None),
        ("new_overhead_dfood_tj_n50_s42_gemini_flash_lite_pre", None),
        ("new_oblique_dfood_tj_n50_s42_gemini_flash_lite_pre", None),
        ("new_both_dfood_tj_n50_s42_gemini_flash_lite_pre", None),
        ("nutrition5k_lite_prep_n50_s42", None),
        ("nutrition5k_lite_preptextcalc_n50_s42", None),
        ("omnifood8k_lite_prep_n50_s42", None),
        ("omnifood8k_lite_preptextcalc_n50_s42", None),
        ("snapme_lite_prep_n50_s42", None),
        ("snapme_lite_preptextcalc_n50_s42", None),
        ("omnifood8k_allimage_density_n50_s42", None),
        ("dens50", None),
        ("new_both_dfood_tj_n350_s42_gemini_3.1_flash_lite_pre", None),
    ]),
    ("Előmérés: szöveges bemenet (\\ref{tab:elomeres_nutribench}.~táblázat) és SNAPMe-ételnévlista", [
        ("nutribench_natural_lite_n50_s42", None),
        ("nutribench_natural_38low_n50_s42", None),
        ("nutribench_natural_gpt-luna_n50_s42", None),
        ("nutribench_natural_sol_n50_s42", None),
        ("nutribench_natural_claude-haiku-low_n50_s42", None),
        ("nutribench_natural_sonnet_n50_s42", None),
        ("nutribench_natural_claude-opus-low_n50_s42", None),
        ("snapme_lite_hint_n50_s42", None),
        ("snapme_sol_hint_n50_s42", None),
        ("snapme_sonnet_hint_n50_s42", None),
    ]),
    ("Egyéb előmérések: megerősítő mérés, reasoning-szint, JPEG-minőség, kalibráció, konzisztencia", [
        ("omnifood8k_lite_macros_confirm_s42", None),
        ("omnifood8k_lite_recipe_confirm_s42", None),
        ("nutrition5k_sol_med_t1000_q80_n50_s42", None),
        ("n5k_lite_macros_q90_n50_s42", None),
        ("n5k_lite_macros_q80_n50_s42", None),
        ("kizart_results/n50_s42_gemini_flash_lite_pre", "omnifood8k"),
        ("kizart_results/n50_s42_gemini_flash_lite_pre", "nutrition5k"),
        ("kizart_results/n50_s42_gemini_flash_lite_pre", "acetada"),
        ("omnifood8k_1image_sol_recipe_n50_s42", None),
    ]),
    ("Döntési mérés", [(f"final_{r}", None) for r in (
        "n5k_lite_macros", "n5k_lite_recipe", "n5k_sonnet_macros", "n5k_sol_macros",
        "omni_lite_macros", "omni_lite_recipe", "omni_sonnet_macros", "omni_sol_macros",
        "omni_sol_recipe",
        "snapme_lite_macros", "snapme_lite_hint",
        "acetada_lite_macros", "acetada_sonnet_macros", "acetada_sol_macros",
        "dfood_lite_macros", "dfood_sonnet_macros", "dfood_sol_macros")]),
]

DATASET_HU = {"nutrition5k": "Nutrition5k", "omnifood8k": "OmniFood8k", "snapme": "SNAPMe",
              "acetada": "ACETADA", "dfood_tj": "DFood-Tj", "nutribench_natural": "NutriBench"}
VIEW_HU = {"overhead": "felülnézet", "oblique": "ferde"}


def run_dir(run: str) -> Path:
    return HERE / run if run.startswith("kizart_results/") else HERE / "results" / run


def settings(entry: dict) -> str:
    out = []
    if entry.get("temperature") is not None:
        out.append(f"temperature {str(entry['temperature']).replace('.', ',')}")
    extra = entry.get("extra") or {}
    if "reasoning_effort" in extra:
        out.append(f"reasoning: {extra['reasoning_effort']}")
    if "effort" in extra:
        out.append(f"effort: {extra['effort']}")
    level = (extra.get("thinking_config") or {}).get("thinking_level")
    if level:
        out.append(f"thinking: {level}")
    tokens = extra.get("max_tokens") or extra.get("max_completion_tokens")
    if tokens:
        out.append(f"max. {tokens} token")
    return ", ".join(out)


def describe(run: str, ds_filter: str | None) -> dict:
    d = run_dir(run)
    recs = [json.loads(line) for line in (d / "results.jsonl").open() if line.strip()]
    recs = [r for r in recs if r.get("final", True)]
    if ds_filter:
        recs = [r for r in recs if r["dataset"] == ds_filter]
    manifests = [json.loads(p.read_text()) for p in sorted(d.glob("manifest_*.json"))]

    def one(key):
        vals = sorted({json.dumps(r.get(key)) for r in recs})
        if len(vals) != 1:
            raise SystemExit(f"{run}: more than one value of {key}: {vals}")
        return json.loads(vals[0])

    label = one("model_label")
    entries = {json.dumps(e, sort_keys=True) for m in manifests for e in m["models"]
               if e["label"] == label}
    if len(entries) != 1:
        raise SystemExit(f"{run}: {len(entries)} different manifest entries for {label}")
    entry = json.loads(entries.pop())
    dataset, modality, n_img = one("dataset"), one("modality"), one("n_images")
    views = one("image_views")
    if n_img is None and modality != "text":
        # first-round records (2026-10-23) have no n_images / image_sizes field: the
        # program then sent one image per call, whose size is in 'image_size'
        if one("image_sizes") is not None or one("image_size") is None:
            raise SystemExit(f"{run}: cannot tell the number of images")
        n_img = 1
    if modality == "text":
        inp = "szöveg"
    else:
        inp = "kép" if n_img == 1 else f"{n_img} kép"
        if dataset == "dfood_tj":
            if views is None:
                dv = {m.get("dfood_views") for m in manifests}
                if len(dv) != 1:
                    raise SystemExit(f"{run}: dfood_views differs between manifests: {dv}")
                views = [dv.pop()]
            inp += " (" + ", ".join(VIEW_HU[v] for v in views) + ")"
        if modality == "image+text":
            inp += " + ételnévlista"
    ts = sorted(r["ts"][:10] for r in recs if r.get("ts"))
    date = ts[0] if ts[0] == ts[-1] else f"{ts[0]}--{ts[-1][5:]}"
    return {"run": run, "date": date,
            "dataset": DATASET_HU[dataset], "model": one("model"),
            "settings": settings(entry), "prompt": one("prompt_variant"), "input": inp,
            "meals": len({r["sample_id"] for r in recs})}


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_")


PROGRAMS = ("benchmark.py", "jpeg80_benchmark.py", "benchmark_new.py",
            "double_call_benchmark.py")


def committed_versions() -> dict[str, tuple[str, str]]:
    """sha256 of every committed version of the benchmark programs -> (program, JPEG)."""
    git = ["git", "-C", str(HERE)]
    commits = subprocess.run(git + ["log", "--format=%H", "--all", "--", *PROGRAMS],
                             check=True, capture_output=True, text=True).stdout.split()
    out = {}
    for c in commits:
        for prog in PROGRAMS:
            r = subprocess.run(git + ["show", f"{c}:{prog}"], capture_output=True)
            if r.returncode != 0:
                continue
            m = re.search(rb"^JPEG_QUALITY = (\d+)", r.stdout, re.M)
            out[hashlib.sha256(r.stdout).hexdigest()] = (prog, m.group(1).decode() if m else "")
    return out


def versions(rows: list[dict], out: Path) -> None:
    known = committed_versions()
    res = []
    for run in dict.fromkeys(r["run"] for r in rows):
        hashes = {json.loads(p.read_text()).get("benchmark_sha256")
                  for p in run_dir(run).glob("manifest_*.json")}
        if None in hashes:
            res.append({"run": run, "sha256": "", "in_repo": "no hash recorded",
                        "program": "", "jpeg_quality": ""})
            continue
        found = [known.get(h) for h in sorted(hashes)]
        res.append({"run": run, "sha256": " ".join(h[:12] for h in sorted(hashes)),
                    "in_repo": "yes" if all(found) else "no",
                    "program": " ".join(sorted({f[0] for f in found if f})),
                    "jpeg_quality": " ".join(sorted({f[1] for f in found if f}))})
    with (out / "program_versions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(res)
    n = len(res)
    yes = sum(r["in_repo"] == "yes" for r in res)
    nohash = sum(r["in_repo"] == "no hash recorded" for r in res)
    fin = [r for r in res if r["run"].startswith("final_")]
    print(f"{n} runs: program version in the repository {yes}, not {n - yes} "
          f"(of these, no hash recorded: {nohash}); final runs in the repository: "
          f"{sum(r['in_repo'] == 'yes' for r in fin)}/{len(fin)}")


def main() -> None:
    out = HERE / "config_table"
    out.mkdir(exist_ok=True)
    rows, seen, lines = [], set(), []
    for title, runs in GROUPS:
        missing = [r for r, f in runs if not run_dir(r).exists()]
        if missing:
            raise SystemExit(f"missing runs: {missing}")
        group = [(r, f) for r, f in runs if (r, f) not in seen]
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for r, f in group:
            seen.add((r, f))
            row = describe(r, f)
            row["group"] = title
            rows.append(row)
            name = tex_escape(row["run"]).replace("\\_", "\\_\\allowbreak{}")
            lines.append(" & ".join([f"\\texttt{{{name}}}", row["date"], row["dataset"],
                                     f"\\texttt{{{row['model']}}}", row["settings"],
                                     tex_escape(row["prompt"]), row["input"],
                                     str(row["meals"])]) + " \\\\")
        lines.append("\\hline")
    with (out / "config_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "config_table.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} rows -> {out}")
    if "--versions" in sys.argv[1:]:
        versions(rows, out)


if __name__ == "__main__":
    main()
