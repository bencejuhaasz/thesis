import json
from pathlib import Path

import numpy as np

import significance as sig

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = [("Nutrition5k", "nutrition5k_macros_n50_s42"), ("OmniFood8k", "omnifood8k_allimage_n50_s42"),
        ("ACETADA", "acetada_lite_macros_n50_s42"), ("DFood-Tj", "dfood_lite_macros_oblique_n50_s42"),
        ("SNAPMe", "snapme_lite_macros_n50_s42")]


def interval(p, E):
    v, se, _ = sig.ratio_stats(np.abs(p - E), E)
    return f"{v:.1f} +- {float(sig._student_t.ppf(0.975, len(E) - 1)) * se:.1f}"


print(f"{'':12s} {'meals':>5s}  {'energy pMAE':14s} {'d_hat * m':14s} {'d * m_hat':14s} "
      f"{'mass bias low 1/3':>17s} {'high 1/3':>9s} {'mass slope':>10s} {'energy bias of d_hat * m':>25s}")
for name, run in RUNS:
    rs = [json.loads(line) for line in (RESULTS / run / "results.jsonl").open() if line.strip()]
    rs = [j for j in rs if j["api_ok"] and j["pred"]]
    kept = [j for j in rs if j["gt"]["kcal"] > 0 and j["gt"]["mass_g"] > 0 and j["pred"]["mass_g"] > 0]
    m = np.array([j["gt"]["mass_g"] for j in kept])
    E = np.array([j["gt"]["kcal"] for j in kept])
    m_hat = np.array([j["pred"]["mass_g"] for j in kept])
    E_hat = np.array([j["pred"]["kcal"] for j in kept])
    d, d_hat = E / m, E_hat / m_hat
    third = np.digitize(m, np.quantile(m, [1 / 3, 2 / 3]))
    low, high = (100 * (m_hat[third == k] - m[third == k]).sum() / m[third == k].sum() for k in (0, 2))
    print(f"{name:12s} {len(kept):5d}  {interval(E_hat, E):14s} {interval(d_hat * m, E):14s} "
          f"{interval(d * m_hat, E):14s} {low:+17.1f} {high:+9.1f} {np.polyfit(m, m_hat, 1)[0]:10.2f} "
          f"{100 * (d_hat * m - E).sum() / E.sum():+25.1f}")
    for j in rs:
        if j not in kept:
            print(f"{'':12s} left out: {j['sample_id']} (reference {j['gt']['kcal']} kcal, {j['gt']['mass_g']} g, "
                  f"estimate {j['pred']['mass_g']} g)")
