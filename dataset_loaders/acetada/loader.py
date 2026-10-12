"""ACETADA loader (Coburn et al., arXiv 2507.07048) — for the public release:

    ACETADA-release/
        ACETADA-HF-dataset.csv     806 rows = 806 meals (before + after photo)
        ACETADA/<participant_id>/  images (3001..3154, 152 participants)

CSV schema (wide):
    participant_id, date, meal_type, food_item_count,
    total_kcal, total_protein_g, total_carbs_g, total_fat_g,
    total_portion_g, total_consumed_g,
    {before,after}_{filename,filepath,date,time,timestamp,gps_latitude,gps_longitude}
    food_item_<n>_{name,before_portion_g,after_portion_g,consumed_g,
                   energy_kcal,protein_g,carbs_g,fat_g}      n = 1..15

Returns (meals, items):
    meals - one row per meal, standard names (kcal, protein_g, carb_g, fat_g,
            served_g, consumed_g), before/after image paths, GPS, timestamp,
            text descriptions, and consistency-check columns.
    items - one row per food item (meal_idx = meals index).

Label semantics (paper + check in __main__): nutrients are for the CONSUMED
amount, while the before photo shows the SERVED amount. Use `description_served`
as the text input for a before photo; `description_consumed` matches the label.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
ITEM_RX = re.compile(r"^food_item_(\d+)_(.+)$")

MEAL_RENAME = {
    "participant_id": "participant", "food_item_count": "n_items_declared",
    "total_kcal": "kcal", "total_protein_g": "protein_g",
    "total_carbs_g": "carb_g", "total_fat_g": "fat_g",
    "total_portion_g": "served_g", "total_consumed_g": "consumed_g",
}
ITEM_RENAME = {"before_portion_g": "served_g", "after_portion_g": "leftover_g",
               "energy_kcal": "kcal", "carbs_g": "carb_g"}
ITEM_FIELDS = ["name", "served_g", "leftover_g", "consumed_g",
               "kcal", "protein_g", "carb_g", "fat_g"]
NUTR = ["kcal", "protein_g", "carb_g", "fat_g"]


def _index_images(img_root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    for dirpath, _, files in os.walk(img_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                idx.setdefault(f, []).append(Path(dirpath) / f)
    return idx


def _resolve(fp, fn, bases: list[Path], idx) -> Path | None:
    """Try filepath relative to the release, then a unique filename match."""
    if isinstance(fp, str) and fp.strip():
        p = Path(fp.strip())
        for cand in ([p] if p.is_absolute() else [b / p for b in bases]):
            if cand.is_file():
                return cand
        fn = fn if isinstance(fn, str) and fn.strip() else p.name
    if isinstance(fn, str) and fn.strip():
        hits = idx.get(Path(fn.strip()).name, [])
        if len(hits) == 1:
            return hits[0]
    return None


def _items_long(df: pd.DataFrame) -> pd.DataFrame:
    cols = [(c, int(m.group(1)), m.group(2)) for c in df.columns
            if (m := ITEM_RX.match(c))]
    parts = []
    for n in sorted({n for _, n, _ in cols}):
        part = pd.DataFrame({f: df[c].to_numpy() for c, k, f in cols if k == n})
        part.insert(0, "item_no", n)
        part.insert(0, "meal_idx", df.index.to_numpy())
        parts.append(part)
    items = pd.concat(parts, ignore_index=True).rename(columns=ITEM_RENAME)
    missing = set(ITEM_FIELDS) - set(items.columns)
    if missing:
        raise ValueError(f"item fields missing: {sorted(missing)}; "
                         f"found {sorted(set(items.columns) - {'meal_idx', 'item_no'})}")
    num = [f for f in ITEM_FIELDS if f != "name"]
    items[num] = items[num].apply(pd.to_numeric, errors="coerce")
    empty = items["name"].isna() & items[num].isna().all(axis=1)
    return items[~empty].sort_values(["meal_idx", "item_no"], ignore_index=True)


def _describe(items: pd.DataFrame, grams: str) -> pd.Series:
    def fmt(g):
        return "; ".join(f"{n} ({w:.0f} g)" if pd.notna(w) else str(n)
                         for n, w in zip(g["name"], g[grams]))
    return items.groupby("meal_idx").apply(fmt, include_groups=False)


def load_acetada(root: str | Path = ".", csv: str | Path | None = None):
    root = Path(root)
    csv = Path(csv) if csv else root / "ACETADA-HF-dataset.csv"
    raw = pd.read_csv(csv)

    need = set(MEAL_RENAME) | {"before_filepath", "before_filename"}
    if missing := need - set(raw.columns):
        raise ValueError(f"{csv.name}: missing columns {sorted(missing)}")

    items = _items_long(raw)

    meal_cols = [c for c in raw.columns if not ITEM_RX.match(c)]
    meals = raw[meal_cols].rename(columns=MEAL_RENAME).copy()

    # ---- images ---------------------------------------------------------
    img_root = root / "ACETADA"
    idx = _index_images(img_root)
    bases = [csv.parent, root, img_root]
    new = {}
    for side in ("before", "after"):
        fp, fn = meals.get(f"{side}_filepath"), meals.get(f"{side}_filename")
        if fp is None and fn is None:
            continue
        fp = fp if fp is not None else pd.Series([None] * len(meals))
        fn = fn if fn is not None else pd.Series([None] * len(meals))
        new[f"{side}_path"] = [_resolve(a, b, bases, idx) for a, b in zip(fp, fn)]
    for side in ("before", "after"):
        ts = f"{side}_timestamp"
        if ts in meals:
            new[f"{side}_dt"] = pd.to_datetime(meals[ts], errors="coerce")

    # ---- per-meal aggregates from items + consistency checks -------------
    g = items.groupby("meal_idx")
    agg = g[["served_g", "leftover_g", "consumed_g", *NUTR]].sum(min_count=1)
    new["n_items"] = g.size().reindex(meals.index, fill_value=0).to_numpy()
    for c in NUTR:
        new[f"{c}_items_sum"] = agg[c].reindex(meals.index).to_numpy()
    new["served_items_sum"] = agg["served_g"].reindex(meals.index).to_numpy()
    new["consumed_items_sum"] = agg["consumed_g"].reindex(meals.index).to_numpy()
    new["description_served"] = _describe(items, "served_g").reindex(meals.index).to_numpy()
    new["description_consumed"] = _describe(items, "consumed_g").reindex(meals.index).to_numpy()

    meals = pd.concat([meals, pd.DataFrame(new, index=meals.index)], axis=1)
    meals["has_before"] = meals["before_path"].notna()
    meals["consumed_fraction"] = (meals["consumed_g"] / meals["served_g"]).where(
        meals["served_g"] > 0)
    atw = 4 * meals["protein_g"] + 4 * meals["carb_g"] + 9 * meals["fat_g"]
    meals["atwater_ratio"] = (atw / meals["kcal"]).where(meals["kcal"] > 0)
    meals["suspect"] = (
        (meals["served_g"] <= 0) | (meals["consumed_g"] < 0)
        | (meals["consumed_g"] > meals["served_g"] + 1)
        | (meals[NUTR] < 0).any(axis=1)
        | (meals["n_items"] != meals["n_items_declared"])
    )

    meals.attrs["unreferenced_images"] = len(
        {p for ps in idx.values() for p in ps}
        - set(meals["before_path"].dropna())
        - set(meals.get("after_path", pd.Series(dtype=object)).dropna()))
    return meals, items


def load_image(meals: pd.DataFrame, i, side: str = "before"):
    from PIL import Image, ImageOps
    p = meals.at[i, f"{side}_path"]
    if p.suffix.lower() == ".heic":
        import pillow_heif
        pillow_heif.register_heif_opener()
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def _maxdiff(a: pd.Series, b: pd.Series) -> str:
    d = (a - b).abs()
    return f"max {d.max():.3g}, >1% off in {(d > 0.01 * a.abs().clip(lower=1)).sum()} meals"


if __name__ == "__main__":
    import sys
    meals, items = load_acetada(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(f"{len(meals)} meals, {len(items)} items, "
          f"{meals.participant.nunique()} participants; before photo found: "
          f"{meals.has_before.sum()}, after photo found: "
          f"{meals['after_path'].notna().sum() if 'after_path' in meals else 0}; "
          f"image files not referenced: {meals.attrs['unreferenced_images']}")
    print(meals["meal_type"].value_counts().to_dict())

    print("\n-- consistency --")
    for c in NUTR:
        print(f"{c:10s} total vs sum(items): {_maxdiff(meals[c], meals[f'{c}_items_sum'])}")
    print(f"served_g   total vs sum(items): {_maxdiff(meals.served_g, meals.served_items_sum)}")
    print(f"consumed_g total vs sum(items): {_maxdiff(meals.consumed_g, meals.consumed_items_sum)}")
    it = items
    print(f"items: consumed = served - leftover: "
          f"{_maxdiff(it.consumed_g, it.served_g - it.leftover_g.fillna(0))}")
    # which mass do the item nutrients belong to? energy density per gram should
    # be the same food property whichever is right; compare scatter of kcal/g.
    ok = (it.kcal > 0) & (it.consumed_g > 0) & (it.served_g > 0) & (it.leftover_g > 0)
    if ok.any():
        r_cons = np.corrcoef(it.kcal[ok], it.consumed_g[ok])[0, 1]
        r_serv = np.corrcoef(it.kcal[ok], it.served_g[ok])[0, 1]
        print(f"items with leftovers ({ok.sum()}): corr(kcal, consumed_g)={r_cons:.3f}, "
              f"corr(kcal, served_g)={r_serv:.3f}  (higher one = mass the label refers to)")
    print(f"n_items == food_item_count: {(meals.n_items == meals.n_items_declared).mean():.1%}")
    print(f"median Atwater/kcal = {meals.atwater_ratio.median():.2f}")
    print(f"suspect meals: {meals.suspect.sum()}  "
          f"(served<=0: {(meals.served_g <= 0).sum()}, consumed>served: "
          f"{(meals.consumed_g > meals.served_g + 1).sum()}, negative nutrient: "
          f"{(meals[NUTR] < 0).any(axis=1).sum()})")
    print(f"consumed fraction: median {meals.consumed_fraction.median():.2f}")
    print()
    print(meals[["kcal", "protein_g", "carb_g", "fat_g", "served_g", "consumed_g"]]
          .describe().round(1))
