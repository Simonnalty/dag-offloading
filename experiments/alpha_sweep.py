# -*- coding: utf-8 -*-
"""
补充实验: E-HEFT 的 alpha 细扫 (时间-能耗权衡曲线的数据来源)

alpha 从 0 到 1 步长 0.1, 每个 alpha 在 20 个随机 DAG(n=50) 上评估,
输出 experiments/alpha_sweep.csv, 供 plots.py 画 Pareto 曲线。
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dag_simulator import gen_dag, simulate
from algo_e_eheft import eheft

SEEDS = 20
N = 50
CSV_PATH = Path(__file__).resolve().parent / "alpha_sweep.csv"

rows = []
for a in range(11):
    alpha = a / 10
    ms, en = [], []
    for seed in range(SEEDS):
        dag = gen_dag(n=N, p_edge=0.15, seed=seed)
        r = simulate(dag, eheft(dag, alpha=alpha))
        ms.append(r["makespan"])
        en.append(r["energy"])
    mean_ms = sum(ms) / SEEDS
    mean_en = sum(en) / SEEDS
    rows.append({"alpha": alpha, "makespan": round(mean_ms, 4),
                 "energy": round(mean_en, 4)})
    print(f"alpha={alpha:.1f}  makespan={mean_ms:.3f}  energy={mean_en:.3f}")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["alpha", "makespan", "energy"])
    writer.writeheader()
    writer.writerows(rows)
print(f"已写入 {CSV_PATH}")
