"""OmniFood8K loader (Yu et al., CVPR 2026).

Layout follows the authors' own dataloader (yudongjian/OmniFood8K-food,
utils/utils_data222.py, class Nutrition8k):

    <root>/
        train_new333.txt           split files (any *train*.txt / *test*.txt
        test_new333.txt            is accepted; or pass them explicitly)
        <data_dir>/<sample>/camera_4.jpg   view the authors train/test on
                           *.jpg / *.png   other views (6 per dish per paper)
    data_dir is "1-data" in the authors' code, "8036" in the public release;
    it is auto-detected (the dir in which the sample paths exist).

Split-file line, whitespace separated (NOTE: order differs from Nutrition5k):
    <sample_relpath> <mass_g> <energy> <protein_g> <fat_g> <carb_g>

ENERGY UNIT: in the public release the energy column is kJ, not kcal
(Atwater 4P+4C+9F over it gives ~0.239 = 1/4.184). energy_unit="auto"
decides from that ratio and refuses to guess if it is neither ~1 nor ~0.239.

No fiber, no text description in the split files.

Returns one DataFrame indexed by sample, with split, the 5 targets,
`image_path` (camera_4.jpg), `views` (all camera_*.jpg found), `n_views`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COLS = ["sample", "mass_g", "energy", "protein_g", "fat_g", "carb_g"]
NUM = ["mass_g", "kcal", "protein_g", "fat_g", "carb_g"]
KJ_PER_KCAL = 4.184
IMG_EXT = {".jpg", ".jpeg", ".png"}
MAIN_VIEW = "camera_4.jpg"


def _find_splits(root: Path) -> dict[str, Path]:
    found = {}
    for split in ("train", "test", "val"):
        hits = sorted(p for p in root.glob(f"*{split}*.txt"))
        if len(hits) > 1:
            raise FileNotFoundError(
                f"several {split} split files in {root}: {[h.name for h in hits]}; "
                "pass splits= explicitly")
        if hits:
            found[split] = hits[0]
    if not found:
        raise FileNotFoundError(f"no *train*.txt / *test*.txt in {root}")
    return found


def _read_split(path: Path, split: str) -> pd.DataFrame:
    rows = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != len(COLS):
            raise ValueError(f"{path.name}:{ln}: expected {len(COLS)} fields, "
                             f"got {len(parts)}: {line!r}")
        rows.append(parts)
    df = pd.DataFrame(rows, columns=COLS)
    df[COLS[1:]] = df[COLS[1:]].astype(float)
    df["split"] = split
    return df


def _view_key(p: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)", p.stem)
    return (int(m.group(1)) if m else 1_000, p.name)


def _atwater_ratio(df: pd.DataFrame, energy: pd.Series) -> pd.Series:
    atw = 4 * df["protein_g"] + 4 * df["carb_g"] + 9 * df["fat_g"]
    return (atw / energy).where(energy > 0)


def _resolve_unit(df: pd.DataFrame, energy_unit: str) -> str:
    if energy_unit in ("kcal", "kJ"):
        return energy_unit
    r = _atwater_ratio(df, df["energy"]).median()
    if 0.8 < r < 1.25:
        return "kcal"
    if 0.8 < r * KJ_PER_KCAL < 1.25:
        return "kJ"
    raise ValueError(f"median Atwater/energy = {r:.3f}: neither kcal (~1) nor kJ "
                     f"(~0.239); column order is probably not {COLS[1:]}")


def _find_data_dir(root: Path, samples: pd.Series) -> Path:
    probe = samples.head(50).tolist()
    cands = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    score = {c: sum((c / s).is_dir() for s in probe) for c in cands}
    best = max(score, key=score.get)
    if score[best] == 0:
        raise FileNotFoundError(
            f"no directory under {root} contains the sample paths "
            f"(e.g. {probe[0]!r}); pass data_dir= explicitly")
    return best


def _views(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.suffix.lower() in IMG_EXT
                   and not p.stem.startswith(("rgb-d", "depth"))), key=_view_key)


def load_omnifood8k(root: str | Path, data_dir: str | None = None,
                    splits: dict[str, str | Path] | None = None,
                    energy_unit: str = "auto") -> pd.DataFrame:
    root = Path(root)
    split_files = ({k: Path(v) for k, v in splits.items()} if splits
                   else _find_splits(root))
    df = pd.concat([_read_split(p, s) for s, p in split_files.items()],
                   ignore_index=True)

    dup = df["sample"].duplicated(keep=False)
    if dup.any():
        overlap = df.loc[dup].groupby("sample")["split"].unique()
        raise ValueError(f"{dup.sum()} rows with repeated sample ids, e.g. "
                         f"{overlap.head().to_dict()}")

    unit = _resolve_unit(df, energy_unit)
    df["energy_unit_in_file"] = unit
    df["kcal"] = df["energy"] / (KJ_PER_KCAL if unit == "kJ" else 1.0)
    df["atwater_ratio"] = _atwater_ratio(df, df["kcal"])

    base = root / data_dir if data_dir else _find_data_dir(root, df["sample"])
    df["sample_dir"] = [base / s for s in df["sample"]]
    df["views"] = [_views(d) for d in df["sample_dir"]]
    df["n_views"] = df["views"].map(len)
    main = [d / MAIN_VIEW for d in df["sample_dir"]]
    df["image_path"] = [m if m in v else (v[0] if v else None)
                        for m, v in zip(main, df["views"])]
    df["has_image"] = df["image_path"].notna()
    df = df.set_index("sample", drop=False)
    df.attrs["data_dir"] = base
    return df


def load_image(df: pd.DataFrame, sample: str, view: int | None = None):
    from PIL import Image, ImageOps
    p = df.at[sample, "image_path"] if view is None else \
        df.at[sample, "sample_dir"] / f"camera_{view}.jpg"
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


if __name__ == "__main__":
    import sys
    df = load_omnifood8k(sys.argv[1])
    print(f"data dir: {df.attrs['data_dir']}; energy column in "
          f"{df.energy_unit_in_file.iloc[0]} (converted to kcal)")
    print(f"{len(df)} samples; {df.has_image.sum()} with an image; "
          f"views per sample: {df.n_views.value_counts().sort_index().to_dict()}")
    print(df.groupby("split").size())
    print(f"median Atwater(4P+4C+9F)/kcal = {df.atwater_ratio.median():.2f}")
    if df.has_image.any():
        ex = df[df.has_image].iloc[0]
        print("example views:", [p.name for p in ex["views"]])
    print(df[NUM].describe().round(1))
