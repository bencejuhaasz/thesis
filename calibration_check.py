from __future__ import annotations

import argparse

import numpy as np

import analyze as A

TARGETS = ["kcal", "mass_g", "carb_g", "protein_g", "fat_g"]


def load(spec: str, target: str, dataset: str | None):
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
    ap = argparse.ArgumentParser()
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
