"""ASA24 portion-size image library loader (NCI, epi.grants.cancer.gov/asa24).

NOT a meal dataset: ~17k PNG reference images of single foods at fixed portion
sizes, linked to USDA FNDDS food codes / portions. Useful for
  * single-food portion-size estimation (image -> grams), and
  * reference images (few-shot / retrieval) for the portion step of a pipeline.
No nutrient labels in the files; nutrients come from FNDDS via food code + grams.

    asa24/
        ASA24_ImageMetadata.xlsx   3 sheets: ReadMe + 2 metadata sheets
        images-asa24-2022.xlsx     images used in ASA24-2022
        Asa24PortionImages/        PNGs (2011-2022)

The sheet/column names are not published on the web page, so this loader:
  * reads every sheet and keeps them in `sheets`,
  * finds the image-file column by checking which values exist on disk,
  * maps food code / description / portion / grams by keyword when unambiguous,
  * prints the ReadMe sheet when run as a script.

Returns (images, sheets): images = the metadata sheet that references image
files, plus `path`, `has_file`, `in_asa24_2022`; sheets = {name: DataFrame}.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

IMG_EXT = {".png", ".jpg", ".jpeg", ".gif"}
KEYS = {
    "food_code":   ("foodcode", "food_code", "fnddscode", "fndds_code"),
    "food_desc":   ("fooddesc", "food_desc", "mainfooddesc", "description"),
    "portion_desc": ("portiondesc", "portion_desc", "portion"),
    "grams":       ("gram", "weight", "_g", "wt"),
}
SPECIFIC_FIRST = ["food_code", "grams", "portion_desc", "food_desc"]


def _norm(s) -> str:
    return re.sub(r"[\s\-\.()\[\]/]+", "_", str(s)).strip("_").lower()


def _index_images(d: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for dirpath, _, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                idx.setdefault(f.lower(), Path(dirpath) / f)
                idx.setdefault(os.path.splitext(f)[0].lower(), Path(dirpath) / f)
    return idx


def _key(v) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if re.fullmatch(r"\d+\.0", s):          # Excel turned a numeric name into float
        s = s[:-2]
    return Path(s).name.lower()


def _stem(v) -> str | None:
    k = _key(v)
    return os.path.splitext(k)[0] if k and os.path.splitext(k)[1] in IMG_EXT else k


def _image_column(df: pd.DataFrame, idx: dict[str, Path], probe: int = 200):
    best, best_rate = None, 0.0
    for c in df.columns:
        vals = df[c].dropna().head(probe)
        if vals.empty:
            continue
        rate = sum(_key(v) in idx for v in vals) / len(vals)
        if rate > best_rate:
            best, best_rate = c, rate
    return (best, best_rate) if best_rate >= 0.2 else (None, best_rate)


def _map(df: pd.DataFrame, skip: set) -> dict[str, str]:
    cm, used = {}, set(skip)
    for tgt in SPECIFIC_FIRST:
        cands = [c for c in df.columns if c not in used
                 and any(k in _norm(c) for k in KEYS[tgt])]
        if tgt == "grams":
            cands = [c for c in cands if pd.api.types.is_numeric_dtype(df[c])]
        if tgt == "food_code":
            cands = [c for c in cands if pd.to_numeric(df[c], errors="coerce")
                     .between(1e7, 1e8 - 1).mean() > 0.5] or cands
        if len(cands) == 1:
            cm[tgt] = cands[0]
            used.add(cands[0])
    return cm


def load_asa24(root: str | Path = "asa24"):
    root = Path(root)
    meta = root / "ASA24_ImageMetadata.xlsx"
    sheets = pd.read_excel(meta, sheet_name=None)
    img_dir = root / "Asa24PortionImages"
    idx = _index_images(img_dir if img_dir.is_dir() else root)
    if not idx:
        raise FileNotFoundError(f"no images under {img_dir} (still zipped?)")

    found = {}
    for name, df in sheets.items():
        if "readme" in _norm(name):
            continue
        col, rate = _image_column(df, idx)
        if col is not None:
            found[name] = (col, rate)
    if not found:
        raise ValueError("no sheet has a column whose values match image files. "
                         + "; ".join(f"{n}: {list(d.columns)}" for n, d in sheets.items()))
    name = max(found, key=lambda n: found[n][1])
    img_col = found[name][0]

    images = sheets[name].copy()
    cm = _map(images, {img_col})
    images = images.rename(columns={v: k for k, v in cm.items()})
    images = images.rename(columns={img_col: "image"})
    images["path"] = [idx.get(_key(v)) for v in images["image"]]
    images["has_file"] = images["path"].notna()

    # other metadata sheet(s): join on a shared column if there is exactly one
    for other, odf in sheets.items():
        if other == name or "readme" in _norm(other):
            continue
        shared = [c for c in odf.columns if c in sheets[name].columns and c != img_col]
        key = [c for c in shared if c in cm.values()] or shared
        if len(key) == 1:
            k = next((t for t, v in cm.items() if v == key[0]), key[0])
            add = odf.rename(columns={key[0]: k})
            add = add.drop_duplicates(subset=[k])
            images = images.merge(add, on=k, how="left", suffixes=("", f"_{other}"))

    # ASA24-2022 subset
    v22 = root / "images-asa24-2022.xlsx"
    if v22.exists():
        s22 = pd.read_excel(v22, sheet_name=None)
        names22: set[str] = set()
        for df in s22.values():
            col, _ = _image_column(df, idx)
            if col is not None:
                names22 |= {_stem(v) for v in df[col].dropna()}
        images["in_asa24_2022"] = images["image"].map(_stem).isin(names22) \
            if names22 else pd.NA

    if "grams" in images:
        images["grams"] = pd.to_numeric(images["grams"], errors="coerce")
    images.attrs.update(sheet=name, image_col=img_col, colmap=cm,
                        n_files=len({p for p in idx.values()}))
    return images, sheets


def load_image(images: pd.DataFrame, i: int):
    from PIL import Image
    return Image.open(images["path"].iloc[i]).convert("RGB")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "asa24"
    images, sheets = load_asa24(root)
    for n, d in sheets.items():
        print(f"== sheet {n!r}: {d.shape}")
        if "readme" in _norm(n):
            with pd.option_context("display.max_colwidth", 200, "display.width", 200):
                print(d.to_string(max_rows=60))
        else:
            print("   columns:", list(d.columns))
    a = images.attrs
    print(f"\nimage table = sheet {a['sheet']!r}, image column {a['image_col']!r}, "
          f"mapped {a['colmap']}")
    print(f"{len(images)} rows, {images.has_file.sum()} resolved to files; "
          f"{a['n_files']} image files on disk")
    if "in_asa24_2022" in images:
        print(f"in ASA24-2022: {images.in_asa24_2022.sum()}")
    if "food_code" in images:
        print(f"distinct food codes: {images.food_code.nunique()}")
    if "grams" in images:
        print(images["grams"].describe().round(1))
    print(images.head(3).T)
