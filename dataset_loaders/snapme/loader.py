from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

NUTRIENTS = {
    "KCAL": "kcal",
    "CARB": "carb_g",
    "PROT": "protein_g",
    "TFAT": "fat_g",
    "FIBE": "fiber_g",
    "SUGR": "sugar_g",
    "FoodAmt": "mass_g",
}

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


ROLE_DIRS = {
    "before": "before_photos", "front": "front_photos",
    "after": "after_photos", "ingredients": "ingredients_photos",
}


def _index_images(root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                idx.setdefault(f, []).append(Path(dirpath) / f)
    return {f: sorted(ps) for f, ps in idx.items()}


def _pick(name: str, roles: list[str], idx: dict[str, list[Path]]) -> Path | None:
    ps = idx.get(name, [])
    if len(ps) <= 1:
        return ps[0] if ps else None
    for r in roles:
        hit = [p for p in ps if p.parent.name == ROLE_DIRS[r]]
        if len(hit) == 1:
            return hit[0]
    raise ValueError(f"{name} exists in {len(ps)} places and none is in "
                     f"{[ROLE_DIRS[r] for r in roles]}: {[str(p) for p in ps]}. "
                     "Point root at snapme_cs_db.")


def _companion(name: str, suffix: str, idx: dict[str, list[Path]]) -> Path | None:
    stem, ext = os.path.splitext(name)
    for cand in (f"{stem}_{suffix}{ext}", f"{name}_{suffix}{ext}",
                 f"{stem}_{suffix}.jpeg", f"{stem}_{suffix}.jpg"):
        if cand in idx:
            return _pick(cand, [suffix], idx)
    return None


def load_snapme(root: str | Path, linkfile: str | Path | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    linkfile = Path(linkfile) if linkfile else _find_linkfile(root)

    items = pd.read_csv(linkfile, low_memory=False).copy()
    missing = {"subject_id", "snapme_study_day", "filename",
               "Food_Description", *NUTRIENTS} - set(items.columns)
    if missing:
        raise ValueError(f"{linkfile}: missing columns {sorted(missing)}")

    items["has_image"] = items["filename"].notna()
    items["occasion"] = items["Occ_Name"].map(OCCASIONS) \
        if "Occ_Name" in items else None

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

    multi = it.groupby("filename")[["subject_id", "snapme_study_day"]].nunique()
    if (multi > 1).any(axis=None):
        bad = multi[(multi > 1).any(axis=1)].index.tolist()
        raise ValueError(f"images linked to several subject-days: {bad[:5]}")

    idx = _index_images(root)
    images["path"] = [_pick(f, ["front", "before"] if p else ["before", "front"], idx)
                      for f, p in zip(images.index, images["packaged"])]
    images["has_file"] = images["path"].notna()
    images["photo_dir"] = [p.parent.name if p is not None else None
                           for p in images["path"]]
    images["companion_path"] = [
        _companion(f, "ingredients" if p else "after", idx)
        for f, p in zip(images.index, images["packaged"])
    ]
    images["name_in_several_dirs"] = [len(idx.get(f, [])) > 1 for f in images.index]
    images.index.name = "filename"
    return items, images


def load_image(images: pd.DataFrame, filename: str):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(images.at[filename, "path"])).convert("RGB")


if __name__ == "__main__":
    import sys
    items, images = load_snapme(*sys.argv[1:3])
    print(f"{len(items)} items ({(~items.has_image).sum()} without photo), "
          f"{len(images)} images ({images.has_file.sum()} found on disk, "
          f"{images.packaged.sum()} packaged), "
          f"{images.subject_id.nunique()} participants")
    print(f"companion photos found: {images.companion_path.notna().sum()}; "
          f"photo folders: {images.photo_dir.value_counts(dropna=False).to_dict()}")
    amb = images[images.name_in_several_dirs]
    if len(amb):
        print(f"{len(amb)} file names exist in several folders; folder chosen by "
              "the packaged flag:")
        print(amb[["packaged", "photo_dir"]].to_string())
    print(images[list(NUTRIENTS.values())].describe().round(1))
