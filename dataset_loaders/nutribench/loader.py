"""NutriBench loader (Hua et al., ICLR 2025) — text-only meal descriptions.

Hugging Face layout (dongx1997/NutriBench), parquet shards:
    v1/wweia_meal_metric-*.parquet   (5,532)   US meals, gram amounts
    v1/wweia_meal_natural-*.parquet  (5,532)   same meals, household units
    v1/who_meal_metric-*.parquet     (527)     FAO/WHO GIFT meals, grams
    v1/who_meal_natural-*.parquet    (266)
    v2/train-*.parquet               (15,617)  24 countries, has
                                               `country`, `serving_type`

Columns: meal_description, carb, protein, fat (g), energy (kcal)
[v2: + country (ISO alpha-3), serving_type ('metric' | 'natural')].
No fiber, no mass, no images.

Returns one DataFrame with columns:
    meal_description, carb_g, protein_g, fat_g, kcal, version,
    source ('wweia' | 'who' | None), serving_type, country, meal_key
`meal_key` groups rows with identical nutrient values (+ country): the
metric and natural phrasings of the same meal share ground truth, so they
are not independent samples — group by it when computing CIs or splitting.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RENAME = {"carb": "carb_g", "protein": "protein_g", "fat": "fat_g",
          "energy": "kcal"}
NUM = list(RENAME.values())


def _read_any(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.suffix == ".parquet":
            frames.append(pd.read_parquet(p))     # needs pyarrow
        elif p.suffix == ".csv":
            frames.append(pd.read_csv(p))
        elif p.suffix in (".jsonl", ".json"):
            frames.append(pd.read_json(p, lines=p.suffix == ".jsonl"))
    return pd.concat(frames, ignore_index=True)


def _files(d: Path, prefix: str) -> list[Path]:
    return sorted(p for p in d.glob(f"{prefix}*")
                  if p.suffix in (".parquet", ".csv", ".jsonl", ".json"))


def load_nutribench(root: str | Path, version: str = "v2") -> pd.DataFrame:
    root = Path(root)
    d = root / version
    if not d.is_dir():
        raise FileNotFoundError(f"{d} not found (expected {root}/v1, {root}/v2)")

    if version == "v1":
        parts = []
        for source in ("wweia", "who"):
            for st in ("metric", "natural"):
                files = _files(d, f"{source}_meal_{st}")
                if not files:
                    raise FileNotFoundError(f"no {source}_meal_{st}* in {d}")
                df = _read_any(files)
                df["source"], df["serving_type"] = source, st
                parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        df["country"] = None
    else:
        files = _files(d, "")
        if not files:
            raise FileNotFoundError(f"no data files in {d}")
        df = _read_any(files)
        df["source"] = None

    missing = {"meal_description", *RENAME} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}; got {list(df.columns)}")

    df = df.rename(columns=RENAME)
    df[NUM] = df[NUM].astype(float)
    df["version"] = version
    df["meal_key"] = df.groupby(
        [df["source"].fillna(""), df["country"].fillna(""),
         *(df[c].round(4) for c in NUM)],
        sort=False).ngroup()
    cols = ["meal_description", *NUM, "version", "source",
            "serving_type", "country", "meal_key"]
    return df[cols]


if __name__ == "__main__":
    import sys
    root = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else "v2"
    nb = load_nutribench(root, version)
    sizes = nb.groupby("meal_key").size()
    print(f"{version}: {len(nb)} descriptions, {nb.meal_key.nunique()} distinct "
          f"nutrient profiles ({(sizes == 2).sum()} appear exactly twice)")
    print(nb.groupby(["serving_type"], dropna=False).size())
    if nb["country"].notna().any():
        print(nb["country"].value_counts().head(10))
    print(nb[NUM].describe().round(1))
