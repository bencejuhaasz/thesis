from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TARGET = "carb_g"
M, ALPHA, POWER, N_FINAL = 8, 0.05, 0.8, 200

RUNS = {
    "nutrition5k": ("nutrition5k_macros_n50_s42", "nutrition5k_sonnet_macros_n50_s42",
                    "nutrition5k_sol_macros_n50_s42"),
    "omnifood8k": ("omnifood8k_allimage_n50_s42", "omnifood8k_1image_sonnet_n50_s42",
                   "omnifood8k_1image_sol_n50_s42"),
    "acetada": ("acetada_lite_macros_n50_s42", "acetada_sonnet_macros_n50_s42",
                "acetada_gpt-sol_macros_n50_s42"),
    "dfood_tj": ("dfood_lite_macros_oblique_n50_s42", "dfood_sonnet_macros_oblique_n50_s42",
                 "dfood_gpt-sol_macros_oblique_n50_s42"),
    "snapme": ("snapme_lite_macros_n50_s42", "snapme_sonnet_macros_n50_s42",
               "snapme_sol_macros_n50_s42"),
}


def load(run: str) -> dict:
    out = {}
    for line in (RESULTS / run / "results.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("parse_ok"):
                out[str(r["sample_id"])] = r
    return out


def main():
    z = norm.ppf(1 - ALPHA / M / 2) + norm.ppf(POWER)
    print(f"z_(1-a'/2) + z_(1-b) = {z:.3f}  (m = {M}, power {POWER:.0%}, n = {N_FINAL}), target {TARGET}\n")
    print(f"{'dataset':12s} {'B':38s} {'n':>3s} {'delta':>7s} {'sigma_d':>8s} {'MDD':>6s}  pass")
    for ds, (A, *Bs) in RUNS.items():
        a = load(A)
        passes = []
        for B in Bs:
            b = load(B)
            us = [u for u in a if u in b and a[u]["gt"].get(TARGET) is not None
                  and abs(a[u]["gt"][TARGET] - b[u]["gt"][TARGET]) < 1e-6]
            t = np.array([a[u]["gt"][TARGET] for u in us])
            pa = np.array([a[u]["pred"][TARGET] for u in us])
            pb = np.array([b[u]["pred"][TARGET] for u in us])
            d = np.abs(pb - t) - np.abs(pa - t)
            delta = 100 * d.sum() / t.sum()
            sigma_d = 100 * d.std(ddof=1) / t.mean()
            mdd = z * sigma_d / math.sqrt(N_FINAL)
            passes.append(abs(delta) >= mdd)
            print(f"{ds:12s} {B:38s} {len(t):3d} {delta:+7.1f} {sigma_d:8.1f} {mdd:6.1f}  {abs(delta) >= mdd}")
        print(f"{'':12s} -> {'IN family A' if any(passes) else 'OUT'}\n")


if __name__ == "__main__":
    main()
