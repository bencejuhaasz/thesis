from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

DISH_COLS = ["dish_id", "total_calories", "total_mass",
             "total_fat", "total_carb", "total_protein"]
INGR_COLS = ["ingr_id", "ingr_name", "grams",
             "calories", "fat", "carb", "protein"]
N_I = len(INGR_COLS)


def _parse_row(row: list[str]) -> tuple[list[str], list[list[str]]]:
    while row and row[-1].strip() == "":
        row = row[:-1]
    n = len(row)
    if (n - 6) % N_I == 0:
        head, rest = row[:6], row[6:]
    elif (n - 7) % N_I == 0:
        head, rest = row[:6], row[7:]
        declared = int(float(row[6]))
        if declared != len(rest) // N_I:
            raise ValueError(f"{row[0]}: num_ingrs={declared}, "
                             f"found {len(rest) // N_I}")
    else:
        raise ValueError(f"{row[0]}: unexpected field count {n}")
    chunks = [rest[i:i + N_I] for i in range(0, len(rest), N_I)]
    return head, chunks


def _read_dish_csv(path: Path, cafe: str):
    dishes, ingrs = [], []
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            head, chunks = _parse_row(row)
            dishes.append(head + [cafe, len(chunks)])
            for c in chunks:
                ingrs.append([head[0]] + c)
    return dishes, ingrs


def _read_splits(root: Path) -> dict[str, set[str]]:
    split_dir = root / "dish_ids" / "splits"
    return {p.stem: {l.strip() for l in p.read_text().splitlines() if l.strip()}
            for p in sorted(split_dir.glob("*.txt"))}


def load_nutrition5k(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    meta = root / "metadata"

    all_d, all_i = [], []
    for cafe in ("cafe1", "cafe2"):
        p = meta / f"dish_metadata_{cafe}.csv"
        if p.exists():
            d, i = _read_dish_csv(p, cafe)
            all_d += d
            all_i += i

    dishes = pd.DataFrame(all_d, columns=DISH_COLS + ["cafe", "num_ingrs"])
    num = DISH_COLS[1:]
    dishes[num] = dishes[num].astype(float)

    ingredients = pd.DataFrame(all_i, columns=["dish_id"] + INGR_COLS)
    num = INGR_COLS[2:]
    ingredients[num] = ingredients[num].astype(float)

    splits = _read_splits(root)
    for name, ids in splits.items():
        dishes[f"in_{name}"] = dishes["dish_id"].isin(ids)
    dishes["split"] = None
    for s in ("train", "test"):
        ids = splits.get(f"rgb_{s}_ids")
        if ids is not None:
            dishes.loc[dishes["dish_id"].isin(ids), "split"] = s

    overhead = root / "imagery" / "realsense_overhead"
    dishes["rgb_path"] = [overhead / d / "rgb.png" for d in dishes["dish_id"]]
    dishes["has_overhead"] = [p.exists() for p in dishes["rgb_path"]]

    if dishes["dish_id"].duplicated().any():
        dup = dishes.loc[dishes["dish_id"].duplicated(), "dish_id"].tolist()
        raise ValueError(f"duplicate dish_ids: {dup[:5]}")

    return dishes.set_index("dish_id", drop=False), ingredients


def load_rgb(dishes: pd.DataFrame, dish_id: str):
    from PIL import Image
    return Image.open(dishes.at[dish_id, "rgb_path"]).convert("RGB")


if __name__ == "__main__":
    import sys
    dishes, ingredients = load_nutrition5k(sys.argv[1])
    print(f"{len(dishes)} dishes, {len(ingredients)} ingredient rows, "
          f"{dishes['has_overhead'].sum()} with overhead RGB")
    print(dishes.groupby("split", dropna=False)["has_overhead"].agg(["size", "sum"]))
    print(dishes[DISH_COLS[1:]].describe().round(1))
