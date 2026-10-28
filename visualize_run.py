#!/usr/bin/env python3
"""
visualize_run.py - one picture per meal: the photo(s) the model got, with the
estimated and true values written onto it, plus the model's item list.

Put it next to benchmark.py. No API calls.
    python visualize_run.py RUN [RUN ...]
    python visualize_run.py nutrition5k_macros_n50_s42 --limit 20          # 20 worst meals
    python visualize_run.py omnifood8k_1image_n50_s42 --sort id

Output: visual/<RUN>/  (a separate folder, next to results/)
    NN_<kcal error>pct_<sample>.jpg   sorted worst first (--sort abs), NN = rank
    index.csv                        the numbers of every picture
    index.html                       open in a browser: all pictures on one page
The photos are found again from the run's own manifest (same loader, colmap, query
and view settings the run used). If a meal's photo count differs from what the
record says the model got, it is reported (catches swapped run folders).
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETS = [("kcal", "kcal", "{:.0f}"), ("mass_g", "g", "{:.0f}"), ("carb_g", "carb g", "{:.1f}"),
           ("protein_g", "protein g", "{:.1f}"), ("fat_g", "fat g", "{:.1f}")]


def load_bench(path: Path):
    spec = importlib.util.spec_from_file_location("bench", path)
    b = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(b)
    return b


def font(size: int, mono: bool = False):
    from PIL import ImageFont
    names = (["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Menlo.ttc", "consola.ttf"] if mono
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf", "Helvetica.ttc"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def paths_for_run(b, run_dir: Path) -> dict:
    """(dataset, sample_id) -> (paths, views), rebuilt with the run's own settings."""
    mans = sorted(run_dir.glob("manifest_*.json"))
    if not mans:
        raise SystemExit(f"{run_dir}: no manifest")
    m = json.loads(mans[-1].read_text())
    for key, glob_name in (("dfood_views", "DFOOD_VIEWS"), ("omnifood_views", "OMNIFOOD_VIEWS"),
                           ("omnifood_min_views", "OMNIFOOD_MIN_VIEWS")):
        if key in m and hasattr(b, glob_name):
            setattr(b, glob_name, m[key])
    if "dfood_send_order" in m and hasattr(b, "DFOOD_SEND_ORDER"):
        b.DFOOD_SEND_ORDER = tuple(m["dfood_send_order"])
    if "dfood_pair_order" in m and hasattr(b, "DFOOD_PAIR_ORDER"):
        b.DFOOD_PAIR_ORDER = tuple(m["dfood_pair_order"])
    out = {}
    for ds, cfg in m["datasets"].items():
        cfg = {k: v for k, v in cfg.items() if k not in ("prompt", "schema", "sample_ids")}
        try:
            root, _, res = b.call_loader(ds, cfg)
            df = b.standardize(ds, cfg, b.pick_frame(ds, cfg, res), root)
        except Exception as e:
            print(f"  [{ds}] cannot reload the dataset: {type(e).__name__}: {e}")
            continue
        want = set(m["datasets"][ds].get("sample_ids", []))
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            if want and sid not in want:
                continue
            out[(ds, sid)] = (b.paths_of(row), b.views_of(row))
    return out


def items_of(text: str) -> list[dict]:
    try:
        obj = json.loads(text)
    except Exception:
        mm = re.search(r"\{.*\}", text or "", re.S)
        try:
            obj = json.loads(mm.group(0)) if mm else {}
        except Exception:
            obj = {}
    its = obj.get("items") if isinstance(obj, dict) else None
    return its if isinstance(its, list) else []


def render(paths, views, rec, items, out_path: Path, height: int):
    from PIL import Image, ImageDraw, ImageOps
    ims = []
    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            ims.append(im.resize((max(1, round(im.width * height / im.height)), height)))
    gap = 6
    width = max(sum(i.width for i in ims) + gap * (len(ims) - 1), 900)
    f_tab, f_small, f_hdr, f_items = font(17, mono=True), font(15), font(16), font(14, mono=True)
    lines = []
    for it in items[:16]:
        def g(k):
            v = it.get(k)
            return f"{v:.0f}" if isinstance(v, (int, float)) else "-"
        lines.append(f"{str(it.get('name', ''))[:44]:44s} {g('estimated_weight_grams'):>5s} g "
                     f"{g('estimated_calories'):>5s} kcal  C {g('estimated_carbohydrate_grams'):>4s} "
                     f"P {g('estimated_protein_grams'):>4s} F {g('estimated_fat_grams'):>4s}")
    if len(items) > 16:
        lines.append(f"... {len(items) - 16} more items")
    strip_h = 34 + 21 * max(1, len(lines))
    canvas = Image.new("RGB", (width, height + strip_h), "white")
    x = 0
    for i, im in enumerate(ims):
        canvas.paste(im, (x, 0))
        if views:
            ImageDraw.Draw(canvas).text((x + 6, height - 22), str(views[i]), fill="white", font=f_small,
                                        stroke_width=2, stroke_fill="black")
        x += im.width + gap

    # value table on a semi-transparent box over the photo
    p, t = rec.get("pred") or {}, rec.get("gt") or {}
    rows = [f"{'':10s}{'model':>9s}{'true':>9s}{'err':>8s}"]
    for key, lab, fmt in TARGETS:
        pv, tv = p.get(key), t.get(key)
        ps = fmt.format(pv) if isinstance(pv, (int, float)) else "-"
        ts = fmt.format(tv) if isinstance(tv, (int, float)) else "-"
        es = f"{100 * (pv - tv) / tv:+.0f}%" if isinstance(pv, (int, float)) and isinstance(tv, (int, float)) and tv else ""
        rows.append(f"{lab:10s}{ps:>9s}{ts:>9s}{es:>8s}")
    if all(isinstance(d.get(k), (int, float)) and d.get("mass_g") for d in (p, t) for k in ("kcal", "mass_g")):
        rows.append(f"{'kcal/g':10s}{p['kcal'] / p['mass_g']:>9.2f}{t['kcal'] / t['mass_g']:>9.2f}")
    title = f"{rec.get('dataset')} | {rec.get('sample_id')} | {rec.get('model_label')} | {rec.get('prompt_variant')}"
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    box_w = 400
    box_h = 30 + 22 * len(rows)
    d.rectangle([8, 8, 8 + box_w, 8 + box_h], fill=(0, 0, 0, 165))
    d.text((16, 12), title[:58], fill=(255, 230, 120, 255), font=f_hdr)
    for i, r in enumerate(rows):
        d.text((16, 36 + 22 * i), r, fill=(255, 255, 255, 255), font=f_tab)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    dd = ImageDraw.Draw(canvas)
    dd.text((8, height + 8), "model's items:", fill="black", font=f_hdr)
    for i, ln in enumerate(lines or ["(no items)"]):
        dd.text((8, height + 32 + 21 * i), ln, fill="black", font=f_items)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--bench", default="benchmark.py")
    ap.add_argument("--out", default="visual")
    ap.add_argument("--limit", type=int, help="only the first N after sorting")
    ap.add_argument("--sort", default="abs", choices=["abs", "signed", "id"],
                    help="abs = largest |kcal error| first, signed = most underestimated first")
    ap.add_argument("--height", type=int, default=480, help="photo height in pixels")
    a = ap.parse_args()
    b = load_bench(HERE / a.bench)

    for run in a.runs:
        run_dir = HERE / "results" / run
        res = run_dir / "results.jsonl"
        if not res.exists():
            print(f"{run}: no results.jsonl")
            continue
        print(f"\n{run}: finding the photos ...")
        pmap = paths_for_run(b, run_dir)
        recs = [json.loads(l) for l in res.open() if l.strip()]
        recs = [r for r in recs if r.get("parse_ok") and r.get("repeat", 0) == 0]

        def kerr(r):
            pv, tv = (r.get("pred") or {}).get("kcal"), (r.get("gt") or {}).get("kcal")
            return 100 * (pv - tv) / tv if pv is not None and tv else 0.0
        if a.sort == "abs":
            recs.sort(key=lambda r: -abs(kerr(r)))
        elif a.sort == "signed":
            recs.sort(key=kerr)
        else:
            recs.sort(key=lambda r: str(r["sample_id"]))
        if a.limit:
            recs = recs[: a.limit]

        out_dir = HERE / a.out / run
        rows, missing, mismatch = [], 0, []
        for rank, r in enumerate(recs, 1):
            key = (r["dataset"], str(r["sample_id"]))
            if key not in pmap or not pmap[key][0]:
                missing += 1
                continue
            paths, views = pmap[key]
            if r.get("n_images") and r["n_images"] != len(paths):
                mismatch.append((r["sample_id"], r["n_images"], len(paths)))
            safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", str(r["sample_id"]))[-60:]
            name = f"{rank:03d}_{kerr(r):+04.0f}pct_{safe}.jpg"
            render(paths, views, r, items_of(r.get("text")), out_dir / name, a.height)
            p, t = r.get("pred") or {}, r.get("gt") or {}
            rows.append({"file": name, "dataset": r["dataset"], "sample_id": r["sample_id"],
                         "n_images": len(paths), "kcal_err_pct": round(kerr(r), 1),
                         **{f"pred_{k}": p.get(k) for k, _, _ in TARGETS},
                         **{f"true_{k}": t.get(k) for k, _, _ in TARGETS}})
        if rows:
            with open(out_dir / "index.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            cards = "\n".join(
                f'<figure><img src="{html.escape(x["file"])}" loading="lazy">'
                f'<figcaption>{html.escape(str(x["sample_id"]))} - kcal error {x["kcal_err_pct"]:+.0f}%</figcaption></figure>'
                for x in rows)
            (out_dir / "index.html").write_text(
                f"<!doctype html><meta charset='utf-8'><title>{html.escape(run)}</title>"
                "<style>body{font-family:sans-serif;margin:16px}figure{margin:0 0 28px}"
                "img{max-width:100%;border:1px solid #ccc}</style>"
                f"<h1>{html.escape(run)}</h1>{cards}")
        print(f"  {len(rows)} pictures -> {out_dir}" + (f"   ({missing} meals: photo not found)" if missing else ""))
        if mismatch:
            print(f"  WARNING: {len(mismatch)} meals have a different photo count than the run record "
                  f"(record vs rebuilt), e.g. {mismatch[:3]}  -> wrong manifest / swapped run folder?")


if __name__ == "__main__":
    main()
