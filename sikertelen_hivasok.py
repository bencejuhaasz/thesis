import json
import statistics
from pathlib import Path

import config_table as C
import significance as sig

HERE = Path(__file__).resolve().parent

FAILED_RUN = "nutrition5k_sol_med_t1000_q80_n50_s42"


def final_records(run, ds_filter=None):
    last = {}
    for line in (C.run_dir(run) / "results.jsonl").open():
        if line.strip():
            j = json.loads(line)
            if ds_filter is None or j["dataset"].startswith(ds_filter):
                last[(j["dataset"], j["model_label"], str(j["sample_id"]), j.get("repeat", 0))] = j
    return list(last.values())


def interval(x, t):
    v, se, _ = sig.ratio_stats(x, t)
    h = float(sig._student_t.ppf(0.975, len(t) - 1)) * se
    return f"{v:.1f} ({v - h:.1f}; {v + h:.1f})"


pre_runs = [r for title, runs in C.GROUPS if title != "Döntési mérés" for r in runs]
calls = failed = fallback = 0
for run, ds in pre_runs:
    rs = final_records(run, ds)
    f = sum(not j["api_ok"] or not j.get("parse_ok") for j in rs)
    fb = sum(bool((j.get("pred") or {}).get("json_fallback")) for j in rs)
    calls, failed, fallback = calls + len(rs), failed + f, fallback + fb
    if f or fb:
        print(f"{run} {ds or ''}: calls {len(rs)}, failed {f}, JSON extracted from text {fb}")
print(f"pre-run rows of the configuration table: {len(pre_runs)}, calls {calls}, failed {failed}, JSON extracted from text {fallback}")

rs = final_records(FAILED_RUN)
bad = [j for j in rs if not j["api_ok"] or not j.get("parse_ok")]
print(f"\n{FAILED_RUN}: calls {len(rs)}, failed {len(bad)}")
for j in bad:
    print(f"  {j['sample_id']}: api_ok {j['api_ok']}, answer length {len(j.get('text') or '')}, "
          f"output tokens {j['usage']['out']}, reasoning tokens {j['usage']['reasoning']}")
print(f"  median output tokens of the successful calls: "
      f"{statistics.median(j['usage']['out'] for j in rs if j.get('parse_ok'))}")
r = rs[0]
d = sig.load_group(C.run_dir(FAILED_RUN) / "results.jsonl", r["dataset"], r["model_label"], r["prompt_variant"],
                   r["output_mode"])
for k in ("carb_g", "kcal"):
    _, x, t = sig.metric_arrays(d, k, "pMAE")
    _, xa, ta = sig.metric_arrays_all(d, k)
    print(f"  {k}: pMAE on the successful calls {interval(x, t)}, on all calls {interval(xa, ta)}")
