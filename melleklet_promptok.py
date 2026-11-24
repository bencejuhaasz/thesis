import argparse
import json
from collections import Counter
from pathlib import Path

import benchmark as B
import double_call_benchmark as D
import recognition as R

ORDER = ["P3", "P3_macros", "NUTRIBENCH", "P3_recipe", "P3_anchor", "P3_portion",
         "P3_check", "P3_recipe_check", "P3_density"]


def texts() -> dict[str, str]:
    out = {k: B.VARIANTS[k]["base"] for k in ORDER}
    out["P3_prep"] = D.VARIANTS["P3_prep"]["base"]
    out["keret_kep"] = B.build_canonical_prompt("{prompt}", "{hint}")
    out["tobbkepes"] = B.MULTIVIEW_NOTE
    out["lekepezo"] = R.PROMPT.format(dictionary="{dictionary}", items="{items}")
    return out


def write(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in texts().items():
        (out_dir / f"{name}.txt").write_text(t.rstrip("\n") + "\n", encoding="utf-8")
        print(f"{name}.txt {len(t.splitlines())} sor")


def check():
    known = {k: v["base"] for k, v in D.VARIANTS.items()}
    known.update({k: v["base"] for k, v in B.VARIANTS.items()})
    counts = Counter()
    for m in sorted(Path("results").glob("*/manifest_*.json")):
        j = json.loads(m.read_text())
        for cfg in j.get("datasets", {}).values():
            v = cfg.get("prompt_variant")
            state = "nincs a kódban" if v not in known else (
                "egyezik" if cfg.get("prompt") == known[v] else "eltér")
            counts[(v, state)] += 1
            if state != "egyezik":
                print(f"{m.parent.name}: {v} {state}")
        if "multiview_note" in j:
            counts[("multiview_note", "egyezik" if j["multiview_note"] == B.MULTIVIEW_NOTE
                    else "eltér")] += 1
    for m in sorted(Path("recognition").glob("manifest*.json")):
        for inv in json.loads(m.read_text()).get("invocations", []):
            counts[("lekepezo", "egyezik" if inv.get("prompt") == R.PROMPT else "eltér")] += 1
    for (v, state), n in sorted(counts.items(), key=str):
        print(f"{v}\t{state}\t{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path, nargs="?")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    if a.out_dir:
        write(a.out_dir)


if __name__ == "__main__":
    main()
