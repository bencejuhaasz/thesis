#!/usr/bin/env python3
"""
calibration_check.py - reproduces the numbers cited in the thesis section
"Az utólagos kalibráció elvetése" (post-hoc calibration and per-target bias).
No API calls. Put it next to analyze.py; it reuses analyze.py's loader, so the
meals, the repeat averaging and the parse filter are exactly the same.

    venv/bin/python calibration_check.py RUN [RUN ...] [--dataset NAME]

RUN = run name under results/ (same spec syntax as analyze.py: NAME[@VARIANT]).
For every run, on the meals with a parsed answer, it prints:
  bias%   100 * sum(pred - true) / sum(true), per target
  kcal pMAE
    raw     no correction
    scale   leave-one-out level correction: pred_i * sum(true_-i) / sum(pred_-i)
            (the same as analyze.py compare --calibrate)
    linear  leave-one-out linear correction: true = a * pred + b, fitted on the
            other meals, applied to meal i
"""
from __future__ import annotations

import argparse

import numpy as np

import analyze as A

TARGETS = ["kcal", "mass_g", "carb_g", "protein_g", "fat_g"]


def load(spec: str, target: str, dataset: str | None):
    """(pred, true) arrays for one target, or None if the run has no such field."""
    try:
        _, vals = A._compare_load(spec, target, dataset)
    except SystemExit:
        return None
    if len(vals) < 3:
        return None
    keys = sorted(vals)
    return (np.array([vals[k][0] for k in keys]),
            np.array([vals[k][1] for k in keys]))


def loo_linear(p, t):
    out = np.empty_like(p)
    for i in range(len(p)):
        m = np.arange(len(p)) != i
        a, b = np.polyfit(p[m], t[m], 1)
        out[i] = a * p[i] + b
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dataset", default=None)
    a = ap.parse_args()

    head = " ".join(f"{'bias%_' + t:>14s}" for t in TARGETS)
    print(f"{'run':55s} {'n':>4s} {head} {'pMAE_raw':>9s} {'pMAE_scale':>10s} {'pMAE_linear':>11s}")
    for spec in a.runs:
        bias, n = [], 0
        for tg in TARGETS:
            d = load(spec, tg, a.dataset)
            if d is None:
                bias.append(f"{'-':>14s}")
                continue
            p, t = d
            bias.append(f"{100 * (p - t).sum() / t.sum():14.1f}")
        d = load(spec, "kcal", a.dataset)
        if d is None:
            print(f"{spec:55s} {'-':>4s} {' '.join(bias)}   (no kcal)")
            continue
        p, t = d
        n = len(t)
        print(f"{spec:55s} {n:4d} {' '.join(bias)} "
              f"{A.pmae(p, t):9.1f} {A.pmae(A._loo_scale(p, t), t):10.1f} "
              f"{A.pmae(loo_linear(p, t), t):11.1f}")


if __name__ == "__main__":
    main()
