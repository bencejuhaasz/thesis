"""DFood-TJ loader (Jin et al., Scientific Data 13:975, 2026; doi 10.1038/s41597-026-07334-9).

Layout per the paper's "Data Records" section (+ the user's directory listing):

    DFoodTJ/
        DFoodTJ-F-Nutr/DFoodTJ-F-Nutr/
            img/                 3,280 meal photos of 91 hospitalized diabetic patients
            truth/truth.xls      per image: image id, food id, carb, fat, protein,
                                 fiber, calorie  ("actual mass values of each
                                 nutrient per image")
            truth/truth_raw.xlsx + patient id, date, Chinese food names, food ids,
                                 measured food weights (nutrients derive from these)
        DFoodTJ-Rec/{train,val,test}/<class>/   284,933 images, 351 classes
        DFoodTJ-Rec/truth_list/{train,val,test}_list.txt
                                 paper: "subfolder number, picture number, category
                                 number"; the released files have 2 fields per line
        DFoodTJ-Nutr/density.xlsx  351 food types: density (g/ml), mass densities
                                 (g/g; kcal/g), volume densities (g/ml; kcal/ml), EN names
        DFoodTJ-Seg.zip          3,932 images + Labelme-derived masks (not loaded here)

The paper does not publish the exact column names or the image file naming,
so columns are matched by keyword (English and Chinese). If a required column
cannot be matched unambiguously, the loader raises and lists the real columns:
then pass `colmap=` explicitly. Units are not stated in Data Records; Table 5
of the paper uses g and kcal. `atwater_ratio` is reported so you can check
kcal vs kJ on the real file.

Requires: pandas, openpyxl (xlsx), xlrd (legacy .xls).

Returns
    load_fnutr -> (images, raw)
        images: one row per image, columns image_id, food_id, carb_g, fat_g,
                protein_g, fiber_g, kcal, path, has_file, atwater_ratio
                (+ any other columns of truth.xls, untouched)
                + from truth_raw: mass_g, n_items_gt, food_names, mass_ok
                  (see _add_mass: mass_g is NaN where the item rows don't
                  reproduce the image's kcal and food ids)
        raw:    truth_raw.xlsx as read (or None), with matched columns renamed
    load_rec    -> one row per image: path, split, class_dir (+ list-file label)
    load_density-> density.xlsx as read, or None if not present
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUTR = ["carb_g", "fat_g", "protein_g", "fiber_g", "kcal"]

# keyword lists, checked against a normalised column name (lowercase, no spaces)
KEYS = {
    "image_id":  ("image", "img", "picture", "pic", "photo", "图片", "图像", "照片"),
    "food_id":   ("food_id", "foodid", "food", "食物编号", "食物id", "食物"),
    "carb_g":    ("carb", "碳水"),
    "fat_g":     ("fat", "脂肪"),
    "protein_g": ("protein", "蛋白"),
    "fiber_g":   ("fiber", "fibre", "纤维"),
    "kcal":      ("calor", "energy", "kcal", "热量", "能量", "卡路里"),
    # truth_raw only
    "patient_id": ("patient", "患者", "病人"),
    "date":       ("date", "time", "日期", "时间"),
    "weight_g":   ("weight", "mass", "gram", "重量", "克"),
    "food_name":  ("name", "名称", "菜名", "食物名"),
}
# specific targets first so that e.g. "food" does not swallow "food weight"
ORDER = ["carb_g", "fat_g", "protein_g", "fiber_g", "kcal", "patient_id",
         "date", "weight_g", "food_name", "image_id", "food_id"]


def _norm(s) -> str:
    return re.sub(r"[\s\-\.()\[\]/（）]+", "_", str(s)).strip("_").lower()


def _match(df: pd.DataFrame, targets: list[str]) -> dict[str, str]:
    """target -> column, only where exactly one unused column matches."""
    cm, used = {}, set()
    for tgt in [t for t in ORDER if t in targets]:
        cands = [c for c in df.columns if c not in used
                 and any(k in _norm(c) for k in KEYS[tgt])]
        if tgt in NUTR + ["weight_g"]:
            cands = [c for c in cands
                     if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
        if len(cands) == 1:
            cm[tgt] = cands[0]
            used.add(cands[0])
    return cm


def _read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except ImportError as e:  # .xls needs xlrd, .xlsx needs openpyxl
        raise ImportError(f"{path.name}: {e}. pip install xlrd openpyxl") from e


def _find(root: Path, pattern: str) -> Path | None:
    hits = sorted(root.rglob(pattern))
    if len(hits) > 1:
        raise FileNotFoundError(f"several {pattern} under {root}: {hits}")
    return hits[0] if hits else None


def _id_key(v) -> str | None:
    """'0042', 42, 42.0, '42.jpg' -> '42' (numeric ids lose leading zeros)."""
    if pd.isna(v):
        return None
    s = Path(str(v).strip()).stem
    if re.fullmatch(r"\d+(\.0+)?", s):
        return str(int(float(s)))
    return s.lower()


def _index_images(d: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for dirpath, _, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                idx.setdefault(_id_key(f), Path(dirpath) / f)
    return idx


def _food_ids(v) -> list[int]:
    """'[0,159,8]' / 3 / '3' -> sorted list of ints."""
    return sorted(int(x) for x in re.findall(r"\d+", str(v)))


def _add_mass(images: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Per-image total food weight from truth_raw.

    truth_raw has one block of rows per image: the first row of a block carries
    the Image ID (and Date), the following item rows of the same image leave
    them empty (merged cells). So: forward-fill the Image ID, group, sum.
    Check per image: the items' kcal must add up to truth.xls kcal and the item
    food ids must equal truth.xls Food ID. Failing images get mass_g = NaN.
    """
    r = raw.copy()
    r["_key"] = r["image_id"].ffill().map(_id_key)
    for c in ("weight_g", "kcal"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    agg = r.groupby("_key").agg(
        mass_g=("weight_g", "sum"),
        n_items_gt=("weight_g", "size"),
        _kcal_items=("kcal", "sum"),
        food_names=("food_name", lambda s: "; ".join(map(str, s))),
        _ids_items=("food_id", lambda s: sorted(int(float(x)) for x in s)),
    )
    images = images.copy()
    images["_key"] = images["image_id"].map(_id_key)
    images = images.merge(agg, left_on="_key", right_index=True, how="left")
    kcal_ok = (images["_kcal_items"] - images["kcal"]).abs() <= \
        (0.05 + 0.001 * images["kcal"].abs())
    ids_ok = [isinstance(b, list) and _food_ids(a) == b
              for a, b in zip(images["food_id"], images["_ids_items"])]
    images["mass_ok"] = kcal_ok & pd.Series(ids_ok, index=images.index)
    images.loc[~images["mass_ok"], "mass_g"] = float("nan")
    images.attrs["mass_check"] = {
        "images": len(images),
        "with_item_rows": int(images["_kcal_items"].notna().sum()),
        "kcal_sum_ok": int(kcal_ok.sum()),
        "food_ids_ok": int(sum(ids_ok)),
        "mass_ok": int(images["mass_ok"].sum()),
        "bad_examples": images.loc[~images["mass_ok"], "image_id"].head(10).tolist(),
    }
    return images.drop(columns=["_key", "_kcal_items", "_ids_items"])


def load_fnutr(root: str | Path, colmap: dict[str, str] | None = None):
    """root = the DFoodTJ dir, or anything containing truth/ and img/."""
    root = Path(root)
    truth = _find(root, "truth.xls") or _find(root, "truth.xlsx")
    if truth is None:
        raise FileNotFoundError(f"no truth.xls under {root}")
    base = truth.parent.parent                       # .../DFoodTJ-F-Nutr
    images = _read_table(truth)

    need = ["image_id", "food_id", *NUTR]
    cm = colmap or _match(images, need)
    missing = [t for t in ["image_id", *NUTR] if t not in cm]
    if missing:
        raise ValueError(f"{truth.name}: could not match {missing}; columns are "
                         f"{list(images.columns)}. Pass colmap={{target: column}}.")
    images = images.rename(columns={v: k for k, v in cm.items()})
    images[NUTR] = images[NUTR].apply(pd.to_numeric, errors="coerce")

    idx = _index_images(base / "img")
    images["path"] = [idx.get(_id_key(v)) for v in images["image_id"]]
    images["has_file"] = images["path"].notna()
    atw = 4 * images["protein_g"] + 4 * images["carb_g"] + 9 * images["fat_g"]
    images["atwater_ratio"] = (atw / images["kcal"]).where(images["kcal"] > 0)
    images.attrs.update(truth=truth, colmap=cm, n_img_files=len(idx),
                        unmatched_files=len(set(idx) - set(images["image_id"].map(_id_key))))

    raw = None
    rawp = _find(base, "truth_raw.xlsx")
    if rawp is not None:
        raw = _read_table(rawp)
        rcm = _match(raw, ["image_id", "food_id", "patient_id", "date",
                           "weight_g", "food_name", *NUTR])
        raw = raw.rename(columns={v: k for k, v in rcm.items()})
        raw.attrs["colmap"] = rcm
        need_raw = {"image_id", "weight_g", "kcal", "food_id", "food_name"}
        if need_raw <= set(raw.columns):
            attrs = dict(images.attrs)
            images = _add_mass(images, raw)
            images.attrs = {**attrs, **images.attrs}
        else:
            print(f"truth_raw: cannot add mass_g, missing {need_raw - set(raw.columns)}")
    return images, raw


def load_rec(root: str | Path) -> pd.DataFrame:
    """One row per image in DFoodTJ-Rec/{train,val,test}/<class>/."""
    rec = Path(root)
    rec = rec if rec.name == "DFoodTJ-Rec" else rec / "DFoodTJ-Rec"
    rows = []
    for split in ("train", "val", "test"):
        d = rec / split
        if not d.is_dir():
            continue
        for cls in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in sorted(cls.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    rows.append((f, split, cls.name))
    if not rows:
        raise FileNotFoundError(f"no images under {rec}/{{train,val,test}}/<class>/")
    df = pd.DataFrame(rows, columns=["path", "split", "class_dir"])
    df["class_id"] = pd.to_numeric(df["class_dir"], errors="coerce").astype("Int64")

    # truth_list/<split>_list.txt. The paper says "subfolder number, picture
    # number, category number" per line, but the released files have 2 fields,
    # so the columns are kept generic (list_0, list_1, ...) and NOT interpreted.
    # Inspect them (see __main__) before using them as labels.
    lists = rec / "truth_list"
    if not lists.is_dir():
        lists = next((p for p in rec.rglob("truth_list") if p.is_dir()), lists)
    if lists.is_dir():
        parts = []
        for split in ("train", "val", "test"):
            p = lists / f"{split}_list.txt"
            if p.exists():
                t = pd.read_csv(p, sep=r"[\s,;]+", header=None, engine="python",
                                dtype=str)
                t.columns = [f"list_{i}" for i in range(t.shape[1])]
                t["split"] = split
                parts.append(t)
        if parts:
            df.attrs["truth_list"] = pd.concat(parts, ignore_index=True)
    return df


def load_density(root: str | Path) -> pd.DataFrame | None:
    p = _find(Path(root), "density.xlsx")
    return None if p is None else _read_table(p)


def load_image(images: pd.DataFrame, i):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(images.at[i, "path"])).convert("RGB")


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    images, raw = load_fnutr(root)
    a = images.attrs
    print(f"F-Nutr: {a['truth']}")
    print(f"  column map: {a['colmap']}")
    print(f"  {len(images)} rows, {images.has_file.sum()} matched to image files "
          f"({a['n_img_files']} files on disk, {a['unmatched_files']} unreferenced)")
    print(f"  median Atwater(4P+4C+9F)/kcal = {images.atwater_ratio.median():.2f}"
          "  (~1 → kcal, ~0.24 → kJ)")
    print(images[NUTR].describe().round(1))
    if "mass_check" in a:
        print(f"  mass_g check: {a['mass_check']}")
        print(images[["mass_g", "n_items_gt"]].describe().round(1))
        print(f"  kcal per gram median: {(images.kcal / images.mass_g).median():.2f}")
    if raw is not None:
        print(f"\ntruth_raw: {raw.shape}, column map {raw.attrs['colmap']}")
        print("  columns:", list(raw.columns))
        if "patient_id" in raw:
            print(f"  patients: {raw['patient_id'].nunique()}")

    try:
        rec = load_rec(root)
        print(f"\nRec: {len(rec)} images, {rec.class_dir.nunique()} classes")
        print(rec.groupby("split").size().to_dict())
        tl = rec.attrs.get("truth_list")
        if tl is not None:
            print(f"truth_list: {len(tl)} lines, {tl.shape[1] - 1} fields per line; "
                  f"per split {tl.groupby('split').size().to_dict()}")
            print(tl.groupby("split").head(2).to_string(index=False))
    except FileNotFoundError as e:
        print(f"\nRec: {e}")

    dens = load_density(root)
    print("\ndensity.xlsx:", "not found" if dens is None
          else f"{dens.shape}, columns {list(dens.columns)}")
