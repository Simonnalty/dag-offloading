# -*- coding: utf-8 -*-
"""
批量实验运行器（课题要求2 "比较分析" 的数据来源）

实验矩阵: 算法 × DAG规模 × 带宽场景 × 多个随机seed
输出:
  experiments/results.csv  每行一条实验记录(可复现所有图表)
  控制台汇总表(均值±标准差、makespan最优次数)

用法:
  python experiments/run_matrix.py           # 全量矩阵(约1分钟)
  python experiments/run_matrix.py --quick   # 冒烟测试(小矩阵)
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dag_simulator
from dag_simulator import gen_dag, simulate
from algo_a_random import random_offload
from algo_b_greedy import greedy_offload
from algo_c_heft import heft
from algo_e_eheft import eheft
from algo_d_dqn import dqn_offload

# ---------------- 实验配置 ----------------
ALGORITHMS = [
    ("all-end",       lambda dag: {i: "end" for i in range(dag["n"])}),
    ("all-edge",      lambda dag: {i: "edge" for i in range(dag["n"])}),
    ("all-cloud",     lambda dag: {i: "cloud" for i in range(dag["n"])}),
    ("A-random",      random_offload),
    ("B-greedy",      greedy_offload),
    ("C-HEFT",        heft),
    ("E-EHEFT-a0.75", lambda dag: eheft(dag, alpha=0.75)),
    ("E-EHEFT-a0.5",  lambda dag: eheft(dag, alpha=0.5)),
    ("E-EHEFT-a0.25", lambda dag: eheft(dag, alpha=0.25)),
    ("D-DQN",         dqn_offload),   # 加载 algo_d_dqn_weights.pth 已训练权重
]
N_SIZES = [20, 50, 100]
BW_SCENES = {"normal": 1.0, "low": 0.5}   # 带宽场景: 正常 / 减半
P_EDGE = 0.15
SEEDS = list(range(10))
CSV_PATH = Path(__file__).resolve().parent / "results.csv"

# 带宽场景通过原地修改共享的 BANDWIDTH 字典实现——
# 所有模块 from dag_simulator import BANDWIDTH 拿到的是同一个 dict 对象,
# 原地改值即全局生效(含各算法内部的传输时间计算), 结束后恢复。
_BW_ORIG = dict(dag_simulator.BANDWIDTH)


def set_bw_scale(scale):
    for k in dag_simulator.BANDWIDTH:
        dag_simulator.BANDWIDTH[k] = _BW_ORIG[k] * scale


def run_matrix(n_sizes, seeds, bw_scenes):
    rows = []
    try:
        for bw_name, scale in bw_scenes.items():
            set_bw_scale(scale)
            for n in n_sizes:
                for seed in seeds:
                    dag = gen_dag(n=n, p_edge=P_EDGE, seed=seed)
                    for algo_name, algo_fn in ALGORITHMS:
                        t0 = time.perf_counter()
                        decision = algo_fn(dag)
                        decision_ms = (time.perf_counter() - t0) * 1000
                        r = simulate(dag, decision)
                        rows.append({
                            "bw": bw_name, "n": n, "seed": seed,
                            "algorithm": algo_name,
                            "makespan": r["makespan"], "energy": r["energy"],
                            "decision_ms": round(decision_ms, 3),
                        })
    finally:
        set_bw_scale(1.0)   # 无论是否异常都恢复, 防止污染后续实验
    return rows


def print_summary(rows):
    # 按 (bw, n, algorithm) 聚合
    groups = {}
    for r in rows:
        groups.setdefault((r["bw"], r["n"], r["algorithm"]), []).append(r)
    # makespan 最优次数: 按 (带宽, 规模) 分组, 组内每个 seed 选出最优算法
    wins = {}
    by_scene = {}
    for r in rows:
        by_scene.setdefault((r["bw"], r["n"], r["seed"]), []).append(r)
    for (bw, n, _seed), rs in by_scene.items():
        best = min(rs, key=lambda r: r["makespan"])["algorithm"]
        wins.setdefault((bw, n), {})
        wins[(bw, n)][best] = wins[(bw, n)].get(best, 0) + 1

    print(f"\n{'带宽':<8}{'规模':>5}  {'算法':<16}"
          f"{'时延均值':>10}{'±std':>8}{'能耗均值':>10}{'±std':>8}"
          f"{'决策耗时ms':>11}{'最优次数':>9}")
    for (bw, n, algo), rs in sorted(groups.items(),
                                    key=lambda x: (x[0][0], x[0][1], x[0][2])):
        ms = [r["makespan"] for r in rs]
        en = [r["energy"] for r in rs]
        dm = [r["decision_ms"] for r in rs]
        print(f"{bw:<8}{n:>5}  {algo:<16}"
              f"{statistics.mean(ms):>10.3f}{statistics.stdev(ms):>8.3f}"
              f"{statistics.mean(en):>10.3f}{statistics.stdev(en):>8.3f}"
              f"{statistics.mean(dm):>11.3f}{wins.get((bw, n), {}).get(algo, 0):>9}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="冒烟测试: 小矩阵")
    args = parser.parse_args()

    n_sizes = [20, 50] if args.quick else N_SIZES
    seeds = list(range(3)) if args.quick else SEEDS

    print(f"实验矩阵: {len(ALGORITHMS)} 算法 × {len(n_sizes)} 规模 "
          f"× {len(BW_SCENES)} 带宽场景 × {len(seeds)} seeds")
    t0 = time.perf_counter()
    rows = run_matrix(n_sizes, seeds, BW_SCENES)
    print(f"共 {len(rows)} 条实验记录, 耗时 {time.perf_counter()-t0:.1f}s")

    CSV_PATH.parent.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果已写入 {CSV_PATH}")

    print_summary(rows)


if __name__ == "__main__":
    main()
