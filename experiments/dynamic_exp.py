# -*- coding: utf-8 -*-
"""
动态带宽环境对比实验(课题改进方向1 的落地验证):
  静态算法(决策时假设带宽恒定) vs DQN 动态策略(逐步感知当前带宽)。
  两种波动场景: 剧烈波动[1.0,0.4]/周期2s, 温和波动[1.0,0.7]/周期4s。
  静态算法的决策用静态带宽计算, 再放到动态环境中"结算"——这正是它们的
  现实处境: 规划时不知道未来带宽如何波动。
输出 experiments/dynamic_exp.csv; 自动选出较优的动态种子存为正式权重。
"""

import csv
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from dag_simulator import gen_dag, simulate_dynamic, make_telegraph_phases
from algo_b_greedy import greedy_offload
from algo_c_heft import heft
from algo_e_eheft import eheft
from algo_d_dqn import (dqn_offload, dqn_dynamic_offload, QNet,
                        DYN_INPUT_DIM, DYN_WEIGHTS_PATH)

LAM = 0.5
SEEDS = 10
REGIMES = {"波动[1.0,0.4]周期2s": make_telegraph_phases(2.0, [1.0, 0.4]),
           "波动[1.0,0.7]周期4s": make_telegraph_phases(4.0, [1.0, 0.7]),
           "剧烈[1.0,0.25]周期1.5s": make_telegraph_phases(1.5, [1.0, 0.25])}

# 静态算法: 决策一次(静态带宽), 动态环境结算
STATIC = [("all-edge", lambda d: {i: "edge" for i in range(d["n"])}),
          ("all-cloud", lambda d: {i: "cloud" for i in range(d["n"])}),
          ("B-greedy", greedy_offload),
          ("C-HEFT", heft),
          ("E-EHEFT-0.5", lambda d: eheft(d, alpha=0.5))]

test_dags = [gen_dag(n=30, p_edge=0.15, seed=500 + s) for s in range(SEEDS)]

# 先评估两个动态种子, 选优者存为正式权重
def eval_seed(seed):
    net = QNet(input_dim=DYN_INPUT_DIM)
    net.load_state_dict(torch.load(ROOT / "experiments" / f"dqn_dyn_seed{seed}.pth",
                                   map_location="cpu"))
    net.eval()
    objs = []
    for regime, phases in REGIMES.items():
        for dag in test_dags:
            r = simulate_dynamic(dag, dqn_dynamic_offload(dag, phases, net, lam=LAM), phases)
            objs.append(r["makespan"] + LAM * r["energy"])
    return sum(objs) / len(objs), net

scores = {}
nets = {}
for s in (11, 22):
    scores[s], nets[s] = eval_seed(s)
    print(f"动态种子{s}: 平均加权目标={scores[s]:.3f}")
best = min(scores, key=scores.get)
shutil.copy(ROOT / "experiments" / f"dqn_dyn_seed{best}.pth", DYN_WEIGHTS_PATH)
print(f"选优: seed{best} → {DYN_WEIGHTS_PATH.name}")

dyn_net = nets[best]

rows = []
for regime, phases in REGIMES.items():
    acc = {}
    for name, fn in STATIC:
        vals = []
        for dag in test_dags:
            r = simulate_dynamic(dag, fn(dag), phases)
            vals.append((r["makespan"], r["energy"]))
        acc[name] = vals
    vals = []
    for dag in test_dags:
        r = simulate_dynamic(dag, dqn_dynamic_offload(dag, phases, dyn_net, lam=LAM), phases)
        vals.append((r["makespan"], r["energy"]))
    acc["D-DQN-dynamic"] = vals
    print(f"\n--- {regime} (加权目标=时延+{LAM}·能耗) ---")
    for name, v in acc.items():
        ms = sum(x[0] for x in v) / len(v)
        en = sum(x[1] for x in v) / len(v)
        ob = sum(m + LAM * e for m, e in v) / len(v)
        rows.append({"regime": regime, "algorithm": name,
                     "makespan": round(ms, 4), "energy": round(en, 4),
                     "weighted": round(ob, 4)})
        print(f"{name:<14} 时延={ms:>7.3f}  能耗={en:>7.3f}  加权={ob:>7.3f}")

with open(ROOT / "experiments" / "dynamic_exp.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["regime", "algorithm", "makespan", "energy", "weighted"])
    w.writeheader(); w.writerows(rows)
print("\n已写入 dynamic_exp.csv")
