import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATASETS = ["n5k", "omni", "acetada", "dfood", "snapme"]
MODELS = ["lite", "sonnet", "sol"]
N5K_GROUPS = ([0, 2, 4, 7, 99], ["1-2", "3-4", "5-7", "8+"])


def span(values, n=1):
    return f"{min(values):.{n}f}-{max(values):.{n}f}"


def final_records(run):
    last = {}
    for line in (RESULTS / run / "results.jsonl").open():
        if line.strip():
            j = json.loads(line)
            last[str(j["sample_id"])] = j
    return last


def output_price(run):
    manifest = json.loads(sorted((RESULTS / run).glob("manifest_*.json"))[-1].read_text())
    return manifest["models"][0]["price"]["out"]


ops = pd.read_csv(HERE / "final_descriptive" / "ops.csv").set_index("run")
print(ops[["calls", "latency_p50_s", "latency_mean_s", "cost_per_1000_usd"]].round(2).to_string())

macros = {(ds, m): ops.loc[f"final_{ds}_{m}_macros"] for ds in DATASETS for m in MODELS
          if f"final_{ds}_{m}_macros" in ops.index}
print()
for m in MODELS:
    rows = [v for (ds, mm), v in macros.items() if mm == m]
    print(f"{m}, P3_macros, {len(rows)} datasets: latency median {span([r.latency_p50_s for r in rows])} s, "
          f"cost {span([r.cost_per_1000_usd for r in rows])} USD/1000")
for m in ("sonnet", "sol"):
    print(f"{m} / lite: latency median "
          f"{span([macros[ds, m].latency_p50_s / macros[ds, 'lite'].latency_p50_s for ds in DATASETS[:4]], 2)}, "
          f"cost {span([macros[ds, m].cost_per_1000_usd / macros[ds, 'lite'].cost_per_1000_usd for ds in DATASETS[:4]], 2)}")
print(f"sol / sonnet: latency median "
      f"{span([macros[ds, 'sol'].latency_p50_s / macros[ds, 'sonnet'].latency_p50_s for ds in DATASETS[:4]], 2)}, "
      f"cost {span([macros[ds, 'sol'].cost_per_1000_usd / macros[ds, 'sonnet'].cost_per_1000_usd for ds in DATASETS[:4]], 2)}")

shares = []
for ds in ("n5k", "dfood"):
    for m in MODELS:
        run = f"final_{ds}_{m}_macros"
        price = output_price(run)
        rs = [j for j in final_records(run).values() if j["api_ok"] and j["usage"]]
        share = 100 * sum(j["usage"]["out"] * price / 1e6 / j["cost_usd"] for j in rs) / len(rs)
        shares.append(share)
        inputs = sorted({j["usage"]["in"] for j in rs})
        print(f"\n{run}: output tokens' share of the cost {share:.1f}%, "
              f"distinct input token counts {inputs if len(inputs) < 4 else len(inputs)}")
print(f"output tokens' share of the cost: {span(shares)}%")

if len(sys.argv) > 1 and sys.argv[1] == "--complexity":
    import benchmark as B
    _, _, out = B.call_loader("nutrition5k", B.DATASETS["nutrition5k"])
    n_ref = out[1].groupby("dish_id").size().astype(int).to_dict()
    bins, labels = N5K_GROUPS
    cost = {}
    for m in MODELS:
        rs = final_records(f"final_n5k_{m}_macros")
        d = pd.DataFrame([dict(sid=k, tout=j["usage"]["out"], cost=j["cost_usd"])
                          for k, j in rs.items() if j["api_ok"] and j["usage"]])
        d["n_ref"] = d.sid.map(n_ref)
        d = d.dropna()
        d["group"] = pd.cut(d.n_ref, bins=bins, labels=labels)
        g = d.groupby("group", observed=True).agg(meals=("sid", "size"), tout_median=("tout", "median"),
                                                  cost_per_1000=("cost", lambda x: 1000 * x.mean()))
        cost[m] = g.cost_per_1000
        print(f"\nfinal_n5k_{m}_macros by the number of reference ingredients\n{g.round(2).to_string()}")
    print(f"cost {labels[-1]} / {labels[0]}: "
          + ", ".join(f"{m} {cost[m][labels[-1]] / cost[m][labels[0]]:.2f}" for m in MODELS))
