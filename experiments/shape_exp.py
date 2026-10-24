# -*- coding: utf-8 -*-
"""
DAG 形状实验(任务3): 深链式 vs 宽扇入 vs 随机, 规模均约 60 节点
  深链式: 20 层 × 3 宽 (视频分析类流水线)
  宽扇入: 5 层 × 12 宽 (工业物联网类汇聚)
  随机:   gen_dag(n=60) 基线
输出 experiments/shape_exp.csv
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dag_simulator import gen_dag, gen_dag_layered, simulate
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
SHAPES = {"deep(20x3)": lambda s: gen_dag_layered(20, 3, seed=s),
          "wide(5x12)": lambda s: gen_dag_layered(5, 12, seed=s),
          "random(60)": lambda s: gen_dag(n=60, p_edge=0.15, seed=s)}
SEEDS = 10

rows = []
for shape_name, gen in SHAPES.items():
    acc = {name: [0.0, 0.0] for name, _ in ALGOS}   # [时延和, 能耗和]
    for seed in range(SEEDS):
        dag = gen(seed)
        for name, fn in ALGOS:
            r = simulate(dag, fn(dag))
            acc[name][0] += r["makespan"]
            acc[name][1] += r["energy"]
    for name, _ in ALGOS:
        rows.append({"shape": shape_name, "algorithm": name,
                     "makespan": round(acc[name][0] / SEEDS, 4),
                     "energy": round(acc[name][1] / SEEDS, 4)})
    print(f"{shape_name}: " + "  ".join(
        f"{n}={acc[n][0]/SEEDS:.2f}s/{acc[n][1]/SEEDS:.1f}J" for n, _ in ALGOS))

with open(ROOT / "experiments" / "shape_exp.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["shape", "algorithm", "makespan", "energy"])
    w.writeheader(); w.writerows(rows)
print("已写入 shape_exp.csv")
