"""SNAPMe loader (Larke et al., Nutrients 2023).

Ground truth is the master ASA24 linkage file: one row per *food item*,
with the photo it appears in (`filename`), FNDDS food code + description,
portion in grams (`FoodAmt`) and the full ASA24 nutrient columns
(KCAL, CARB, PROT, TFAT, FIBE, SUGR, ...), already scaled to the portion.

Returns:
    items  - one row per food item (all 121 ASA24 columns kept)
    images - one row per before/front photo: summed nutrients, a text
             description built from the items, and paths to the photo and
             its companion (_after / _ingredients) photo if present.

Notes on the labels (matter for evaluation):
  * Portions are the participant's own ASA24 report, not weighed.
  * The record describes what is in the "before"/"front" photo; "after"
    photos (leftovers) carry no nutrients of their own.
  * A few item rows have no photo (filename NaN); they stay in `items`
    with has_image=False and are excluded from `images`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ASA24 column -> friendly name for the per-image table
NUTRIENTS = {
    "KCAL": "kcal",
    "CARB": "carb_g",
    "PROT": "protein_g",
    "TFAT": "fat_g",
    "FIBE": "fiber_g",
    "SUGR": "sugar_g",
    "FoodAmt": "mass_g",
}

# ASA24 eating-occasion codes (Occ_Name)
OCCASIONS = {1: "breakfast", 2: "brunch", 3: "lunch", 4: "dinner",
             5: "supper", 6: "snack", 7: "just a drink", 8: "just a supplement"}

IMG_EXT = {".jpeg", ".jpg", ".png"}


def _find_linkfile(root: Path) -> Path:
    hits = sorted(p for p in root.rglob("*.csv") if "linkfile" in p.name.lower())
    master = [p for p in hits if "master" in p.name.lower()]
    hits = master or hits
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected one master linkage CSV under {root}, found: {hits}. "
            "Pass linkfile= explicitly.")
    return hits[0]


def _index_images(root: Path) -> dict[str, Path]:
    """filename -> path for every image under root (first hit wins)."""
    idx: dict[str, Path] = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                idx.setdefault(f, Path(dirpath) / f)
    return idx


def _companion(name: str, suffix: str, idx: dict[str, Path]) -> Path | None:
    # Paper: "{before_filename}_after.jpeg"; accept both readings of that.
    stem, ext = os.path.splitext(name)
    for cand in (f"{stem}_{suffix}{ext}", f"{name}_{suffix}{ext}",
                 f"{stem}_{suffix}.jpeg", f"{stem}_{suffix}.jpg"):
        if cand in idx:
            return idx[cand]
    return None


def load_snapme(root: str | Path, linkfile: str | Path | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    linkfile = Path(linkfile) if linkfile else _find_linkfile(root)

    items = pd.read_csv(linkfile, low_memory=False).copy()  # copy: consolidate blocks
    missing = {"subject_id", "snapme_study_day", "filename",
               "Food_Description", *NUTRIENTS} - set(items.columns)
    if missing:
        raise ValueError(f"{linkfile}: missing columns {sorted(missing)}")

    items["has_image"] = items["filename"].notna()
    items["occasion"] = items["Occ_Name"].map(OCCASIONS) \
        if "Occ_Name" in items else None

    # ---- per-image ground truth --------------------------------------
    it = items[items["has_image"]]
    agg = it.groupby("filename").agg(
        subject_id=("subject_id", "first"),
        study_day=("snapme_study_day", "first"),
        packaged=("packaged_food", "max"),
        n_items=("Food_Description", "size"),
        occasion=("occasion", "first"),
    )
    sums = it.groupby("filename")[list(NUTRIENTS)].sum().rename(columns=NUTRIENTS)
    desc = it.groupby("filename").apply(
        lambda g: "; ".join(f"{d} ({a:.0f} g)"
                            for d, a in zip(g["Food_Description"], g["FoodAmt"])),
        include_groups=False,
    ).rename("description")
    images = agg.join(sums).join(desc)
    images["packaged"] = images["packaged"].fillna(0).astype(bool)

    # Sanity check: an image must belong to exactly one participant-day.
    multi = it.groupby("filename")[["subject_id", "snapme_study_day"]].nunique()
    if (multi > 1).any(axis=None):
        bad = multi[(multi > 1).any(axis=1)].index.tolist()
        raise ValueError(f"images linked to several subject-days: {bad[:5]}")

    # ---- image paths ---------------------------------------------------
    idx = _index_images(root)
    images["path"] = [idx.get(f) for f in images.index]
    images["has_file"] = images["path"].notna()
    images["companion_path"] = [
        _companion(f, "ingredients" if p else "after", idx)
        for f, p in zip(images.index, images["packaged"])
    ]
    images.index.name = "filename"
    return items, images


def load_image(images: pd.DataFrame, filename: str):
    from PIL import Image, ImageOps
    # Phone photos: apply EXIF orientation, or many will be sideways.
    return ImageOps.exif_transpose(Image.open(images.at[filename, "path"])).convert("RGB")


if __name__ == "__main__":
    import sys
    items, images = load_snapme(*sys.argv[1:3])
    print(f"{len(items)} items ({(~items.has_image).sum()} without photo), "
          f"{len(images)} images ({images.has_file.sum()} found on disk, "
          f"{images.packaged.sum()} packaged), "
          f"{images.subject_id.nunique()} participants")
    print(f"companion photos found: {images.companion_path.notna().sum()}")
    print(images[list(NUTRIENTS.values())].describe().round(1))
