#!/usr/bin/env python3
"""
export_n5k_subset.py - temporary helper: zip the first N dishes of the seeded
Nutrition5k selection (same selection code as benchmark.py), with all their images
and records. The first 50 are exactly the dishes of the n=50, seed=42 runs.

Put it next to benchmark.py.
    venv/bin/python export_n5k_subset.py                 # n=200, seed=42
    venv/bin/python export_n5k_subset.py --n 200 --seed 42 --videos   # + side-angle videos (large)

Output: nutrition5k_first200_s42.zip, unpacks to nutrition5k_first200_s42/ :
    selection.csv          rank (1 = first), dish_id, in_first50
    dishes.csv             the loader's dish records (labels) for these dishes
    ingredients.csv        the loader's ingredient records for these dishes
    files/...              every file of each dish's overhead folder (rgb, depth ...),
                           extracted side frames if any (and videos with --videos),
                           at their path relative to the Nutrition5K folder
    README.txt
"""
import argparse
import csv
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_bench():
    spec = importlib.util.spec_from_file_location("bench", HERE / "benchmark.py")
    b = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE))
    spec.loader.exec_module(b)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--videos", action="store_true", help="also the side-angle .h264 videos (large)")
    ap.add_argument("--check-run", default="nutrition5k_macros_n50_s42",
                    help="run whose manifest must equal the first 50 (skip if missing)")
    a = ap.parse_args()

    b = load_bench()
    if hasattr(b, "N5K_VIEWS"):
        b.N5K_VIEWS = "overhead"                     # one row per dish, overhead rgb path
    cfg = dict(b.DATASETS["nutrition5k"])
    cfg["select"] = {"mode": "random", "n": a.n, "seed": a.seed}
    root, _, out = b.call_loader("nutrition5k", cfg)
    root = Path(root)
    sel = b.standardize("nutrition5k", cfg, b.pick_frame("nutrition5k", cfg, out), root)
    sel = b.select_rows(sel, cfg["select"])
    ids = [str(s) for s in sel["sample_id"]]
    print(f"{len(ids)} dishes selected (seed {a.seed})")

    # the first 50 must be the measured ones
    first50 = set(ids[:50])
    mans = sorted((HERE / "results" / a.check_run).glob("manifest_*.json"))
    if mans:
        m = json.loads(mans[-1].read_text())
        measured = [str(s) for s in m["datasets"]["nutrition5k"]["sample_ids"]]
        same = set(measured) == first50
        print(f"first 50 == {a.check_run}: {'YES' if same else 'NO'}")
        if not same:
            raise SystemExit("selection differs from the measured run - not exporting")
    else:
        print(f"(no manifest for {a.check_run}, first-50 check skipped)")

    dishes = out[0] if isinstance(out, (tuple, list)) else out
    ing = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
    id_col = (cfg.get("colmap") or {}).get("sample_id", "dish_id")
    rank = {d: i for i, d in enumerate(ids, 1)}

    name = f"nutrition5k_first{a.n}_s{a.seed}"
    zpath = HERE / f"{name}.zip"
    files, missing = [], 0
    for _, row in sel.iterrows():
        img = row.get("image_path")
        if not img or not Path(img).is_file():
            missing += 1
            continue
        files += [p for p in Path(img).parent.iterdir() if p.is_file()]      # overhead folder
        side = HERE / getattr(b, "N5K_SIDE_DIR", "Nutrition5K/imagery/side_frames") / str(row["sample_id"])
        if side.is_dir():
            files += [p for p in side.iterdir() if p.is_file()]
        if a.videos:
            vids = list(root.rglob(f"side_angles/{row['sample_id']}/*.h264"))
            files += vids
    if missing:
        print(f"WARNING: {missing} dishes without an overhead image file")

    def arc(p: Path) -> str:
        p = p.resolve()
        for base in (root, HERE):
            try:
                return f"{name}/files/{p.relative_to(base.resolve()).as_posix()}"
            except ValueError:
                pass
        return f"{name}/files/other/{p.name}"

    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        rows = [["rank", "dish_id", "in_first50"]] + [[r, d, int(r <= 50)] for d, r in rank.items()]
        z.writestr(f"{name}/selection.csv", "\n".join(",".join(map(str, r)) for r in rows) + "\n")
        dsub = dishes[dishes[id_col].astype(str).isin(rank)].copy()
        dsub.insert(0, "rank", dsub[id_col].astype(str).map(rank))
        for c in dsub.columns:                       # local absolute paths -> relative
            dsub[c] = dsub[c].map(lambda v: v.replace(str(root) + "/", "") if isinstance(v, str) else v)
        z.writestr(f"{name}/dishes.csv", dsub.sort_values("rank").to_csv(index=False))
        if ing is not None and id_col in ing.columns:
            z.writestr(f"{name}/ingredients.csv",
                       ing[ing[id_col].astype(str).isin(rank)].to_csv(index=False))
        z.writestr(f"{name}/README.txt",
                   f"First {len(ids)} dishes of the Nutrition5k selection used in the benchmark "
                   f"(benchmark.py select: random, seed {a.seed}; same query as the runs).\n"
                   f"Ranks 1-50 = the dishes of the n=50 runs.\n"
                   "selection.csv: order; dishes.csv / ingredients.csv: the dataset's records;\n"
                   "files/: the dishes' image files at their path inside the Nutrition5K folder.\n")
        for p in sorted(set(files)):
            z.write(p, arc(p), compress_type=zipfile.ZIP_STORED)   # images are compressed already
    size = zpath.stat().st_size / 1e6
    print(f"{len(set(files))} files -> {zpath}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
