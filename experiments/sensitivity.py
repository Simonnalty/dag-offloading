# -*- coding: utf-8 -*-
"""
敏感性分析实验(任务2):
  Part A 带宽扫描: 带宽 ×{0.3,0.5,0.7,1.0}, 观察各算法时延如何随传输能力变化
  Part B λ 扫描:   同一批运行数据上计算不同能耗权重下的加权目标
输出 experiments/bw_sweep.csv 与 experiments/lambda_sweep.csv (plots.py 消费)
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dag_simulator
from dag_simulator import gen_dag, simulate
from algo_b_greedy import greedy_offload
from algo_c_heft import heft
from algo_e_eheft import eheft
from algo_d_dqn import dqn_offload

ALGOS = [("all-edge", lambda d: {i: "edge" for i in range(d["n"])}),
         ("all-cloud", lambda d: {i: "cloud" for i in range(d["n"])}),
         ("B-greedy", greedy_offload),
         ("C-HEFT", heft),
         ("E-EHEFT-0.5", lambda d: eheft(d, alpha=0.5)),
         ("D-DQN", dqn_offload)]
SEEDS = 10
N = 50

def run_once():
    """跑一遍全部算法, 返回 {algo: [(makespan, energy), ...]}"""
    out = {name: [] for name, _ in ALGOS}
    for seed in range(SEEDS):
        dag = gen_dag(n=N, p_edge=0.15, seed=seed)
        for name, fn in ALGOS:
            r = simulate(dag, fn(dag))
            out[name].append((r["makespan"], r["energy"]))
    return out

# ---- Part A: 带宽扫描(原地改共享 BANDWIDTH 字典, 结束恢复) ----
ORIG = dict(dag_simulator.BANDWIDTH)
rows_a = []
for scale in (1.0, 0.7, 0.5, 0.3):
    for k in dag_simulator.BANDWIDTH:
        dag_simulator.BANDWIDTH[k] = ORIG[k] * scale
    res = run_once()
    for name, runs in res.items():
        ms = sum(x[0] for x in runs) / SEEDS
        en = sum(x[1] for x in runs) / SEEDS
        rows_a.append({"bw_scale": scale, "algorithm": name,
                       "makespan": round(ms, 4), "energy": round(en, 4)})
    print(f"带宽×{scale}: " + "  ".join(f"{n}={sum(x[0] for x in res[n])/SEEDS:.2f}s" for n, _ in ALGOS))
for k in dag_simulator.BANDWIDTH:
    dag_simulator.BANDWIDTH[k] = ORIG[k]

with open(ROOT / "experiments" / "bw_sweep.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["bw_scale", "algorithm", "makespan", "energy"])
    w.writeheader(); w.writerows(rows_a)

# ---- Part B: λ 扫描(同一批静态运行数据, 对不同 λ 计算加权目标) ----
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
res = run_once()   # 正常带宽
rows_b = []
for lam in LAMBDAS:
    for name, runs in res.items():
        obj = sum(m + lam * e for m, e in runs) / SEEDS
        rows_b.append({"lambda": lam, "algorithm": name, "weighted_obj": round(obj, 4)})
    line = "  ".join(f"{n}={sum(m+lam*e for m,e in res[n])/SEEDS:.2f}" for n, _ in ALGOS)
    print(f"λ={lam}: {line}")

with open(ROOT / "experiments" / "lambda_sweep.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["lambda", "algorithm", "weighted_obj"])
    w.writeheader(); w.writerows(rows_b)

print("已写入 bw_sweep.csv 与 lambda_sweep.csv")
