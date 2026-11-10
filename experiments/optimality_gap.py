# -*- coding: utf-8 -*-
"""
最优性差距实验: 用精确基准量化各启发式离最优有多远 (课题要求2 的补充)

背景: 原实验矩阵只有 all-end/all-edge/all-cloud 三个参照, 它们不是最优解;
      "HEFT 是质量标杆"因此只有相对意义, 没有绝对标尺。本实验在小规模
      DAG 上穷举全部 3^V 个决策 (algo_f_exact.py), 得到:
        · 精确最优解 (λ=0.5 的加权目标)
        · 真实 Pareto 前沿
      再计算每种算法/参照方案:
        gap_pct     = (算法加权目标 / 最优加权目标 - 1) × 100
        ms_ratio    = 算法 makespan / 最优解 makespan
        en_ratio    = 算法能耗 / 最优解能耗
        dominated   = 是否被某个精确前沿点严格支配
        on_front    = 是否恰好落在前沿上 (差距 < 0.05%)

规模选择: n=8/10/12 —— 3^V 分别 6.5K/59K/531K, 单实例 0.05/0.5/5 s;
          n<8 时启发式几乎总能撞上最优, 区分度不足; n>12 枚举不可行。

输出:
  experiments/optimality_gap.csv      每条 (规模, seed, 算法) 记录
  experiments/pareto_front.csv        每个实例的真实前沿点(作图用)
  experiments/optimality_summary.csv  按规模的聚合指标(报告表格用)
  experiments/optimality_gap_case.csv 最差实例的逐方案明细(案例研究用)

用法:
  python experiments/optimality_gap.py          # 全量 (约 1 分钟)
  python experiments/optimality_gap.py --quick  # 冒烟 (n=8, 3 seeds)
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dag_simulator import gen_dag, simulate
from algo_a_random import random_offload
from algo_b_greedy import greedy_offload
from algo_c_heft import heft
from algo_e_eheft import eheft
from algo_f_exact import exact_front, objective

try:                       # DQN 需要 torch; 未安装时自动跳过, 不阻塞其余算法
    from algo_d_dqn import dqn_offload
    HAS_DQN = True
    DQN_SKIP_REASON = ""
except Exception as exc:   # pragma: no cover - 取决于运行环境
    HAS_DQN = False
    DQN_SKIP_REASON = f"torch/DQN 不可用: {exc.__class__.__name__}"

P_EDGE = 0.15
LAM = 0.5
SIZES = {8: 10, 10: 10, 12: 5}     # n -> seeds 数 (枚举成本随 3^n 增长)

ALGORITHMS = [
    ("all-end",        lambda d: {i: "end" for i in range(d["n"])}),
    ("all-edge",       lambda d: {i: "edge" for i in range(d["n"])}),
    ("all-cloud",      lambda d: {i: "cloud" for i in range(d["n"])}),
    ("A-random",       random_offload),
    ("B-greedy",       greedy_offload),
    ("C-HEFT",         heft),
    ("E-EHEFT-a0.25",  lambda d: eheft(d, alpha=0.25)),
    ("E-EHEFT-a0.5",   lambda d: eheft(d, alpha=0.5)),
    ("E-EHEFT-a0.75",  lambda d: eheft(d, alpha=0.75)),
    ("E-EHEFT-a1.0",   lambda d: eheft(d, alpha=1.0)),
]
if HAS_DQN:
    ALGORITHMS.append(("D-DQN", dqn_offload))

GAP_COLS = ["n", "seed", "algorithm", "makespan", "energy", "objective",
            "opt_makespan", "opt_energy", "opt_objective", "gap_pct",
            "ms_ratio", "en_ratio", "min_makespan", "min_energy",
            "ms_ratio_min", "en_ratio_min",
            "dominated", "on_front", "front_size", "points",
            "n_end", "n_edge", "n_cloud", "decision_ms"]
FRONT_COLS = ["n", "seed", "idx", "makespan", "energy",
              "n_end", "n_edge", "n_cloud"]
SUMMARY_COLS = ["n", "algorithm", "instances", "mean_gap_pct", "median_gap_pct",
                "p90_gap_pct", "max_gap_pct", "mean_ms_ratio", "mean_en_ratio",
                "mean_ms_ratio_min", "mean_en_ratio_min",
                "on_front_rate", "dominated_rate", "worst_seed"]


def pct(values, q):
    """无插值分位数(样本少时更稳健)"""
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def run_size(n, seeds):
    """跑一个规模的全部 seed, 返回 (gap_rows, front_rows, case_row)"""
    gap_rows, front_rows = [], []
    case = None
    for seed in seeds:
        dag = gen_dag(n=n, p_edge=P_EDGE, seed=seed)
        front = exact_front(dag)
        opt = min(front, key=lambda p: (p["objective"], p["makespan"], p["energy"]))
        assert abs(objective(opt["makespan"], opt["energy"], LAM)
                   - opt["objective"]) < 1e-3
        min_makespan = front[0]["makespan"]      # 前沿左端 = 理论最快
        min_energy = front[-1]["energy"]         # 前沿右端 = 理论最省电

        for idx, p in enumerate(front):
            front_rows.append({"n": n, "seed": seed, "idx": idx,
                               "makespan": p["makespan"], "energy": p["energy"],
                               "n_end": p["per_node"]["end"],
                               "n_edge": p["per_node"]["edge"],
                               "n_cloud": p["per_node"]["cloud"]})

        per_algo = []
        for name, fn in ALGORITHMS:
            t0 = time.perf_counter()
            decision = fn(dag)
            cost_ms = (time.perf_counter() - t0) * 1000.0
            r = simulate(dag, decision)
            f = objective(r["makespan"], r["energy"], LAM)
            gap = 100.0 * (f / opt["objective"] - 1.0)
            row = {
                "n": n, "seed": seed, "algorithm": name,
                "makespan": r["makespan"], "energy": r["energy"],
                "objective": round(f, 4),
                "opt_makespan": opt["makespan"], "opt_energy": opt["energy"],
                "opt_objective": opt["objective"],
                "gap_pct": round(gap, 3),
                "ms_ratio": round(r["makespan"] / opt["makespan"], 4),
                "en_ratio": round(r["energy"] / opt["energy"], 4),
                "min_makespan": min_makespan, "min_energy": min_energy,
                "ms_ratio_min": round(r["makespan"] / min_makespan, 4),
                "en_ratio_min": round(r["energy"] / min_energy, 4),
                "dominated": int(any(
                    p["makespan"] <= r["makespan"] + 1e-9
                    and p["energy"] <= r["energy"] + 1e-9
                    and (p["makespan"] < r["makespan"] - 1e-9
                         or p["energy"] < r["energy"] - 1e-9) for p in front)),
                "on_front": int(any(
                    abs(p["makespan"] - r["makespan"]) <= 5e-4
                    and abs(p["energy"] - r["energy"]) <= 5e-4 for p in front)),
                "front_size": len(front), "points": 3 ** n,
                "n_end": r["per_node"]["end"], "n_edge": r["per_node"]["edge"],
                "n_cloud": r["per_node"]["cloud"],
                "decision_ms": round(cost_ms, 3),
            }
            gap_rows.append(row)
            per_algo.append(row)

        # 案例研究: 取 HEFT 差距最大的实例(报告里用于解释"差在哪")
        heft_row = next(r for r in per_algo if r["algorithm"] == "C-HEFT")
        if case is None or heft_row["gap_pct"] > case["heft_gap"]:
            case = {"n": n, "seed": seed, "heft_gap": heft_row["gap_pct"],
                    "opt": opt, "front": front, "rows": per_algo}
        print(f"  n={n:>2} seed={seed}: 前沿 {len(front):>2} 点 | "
              + "  ".join(f"{r['algorithm']}={r['gap_pct']:.1f}%"
                          for r in per_algo if r["algorithm"] in
                          ("C-HEFT", "E-EHEFT-a0.5")))
    return gap_rows, front_rows, case


def summarize(rows):
    """按 (n, algorithm) 聚合; 另加两行 oracle: 最佳单算法 / 最佳 α"""
    groups = {}
    for r in rows:
        groups.setdefault((r["n"], r["algorithm"]), []).append(r)
    out = []
    for (n, algo), rs in sorted(groups.items()):
        gaps = [r["gap_pct"] for r in rs]
        out.append({
            "n": n, "algorithm": algo, "instances": len(rs),
            "mean_gap_pct": round(statistics.mean(gaps), 3),
            "median_gap_pct": round(statistics.median(gaps), 3),
            "p90_gap_pct": round(pct(gaps, 0.9), 3),
            "max_gap_pct": round(max(gaps), 3),
            "mean_ms_ratio": round(statistics.mean(r["ms_ratio"] for r in rs), 4),
            "mean_en_ratio": round(statistics.mean(r["en_ratio"] for r in rs), 4),
            "mean_ms_ratio_min": round(statistics.mean(r["ms_ratio_min"] for r in rs), 4),
            "mean_en_ratio_min": round(statistics.mean(r["en_ratio_min"] for r in rs), 4),
            "on_front_rate": round(statistics.mean(r["on_front"] for r in rs), 3),
            "dominated_rate": round(statistics.mean(r["dominated"] for r in rs), 3),
            "worst_seed": rs[gaps.index(max(gaps))]["seed"],
        })

    # 两个"事后诸葛亮"参照, 说明即使把 α/算法选到最好, 离最优仍有差距
    by_n_seed = {}
    for r in rows:
        by_n_seed.setdefault((r["n"], r["seed"]), []).append(r)
    for tag, pick in (
        ("oracle-best-single", lambda rs: min(
            (r for r in rs if not r["algorithm"].startswith("E-EHEFT")),
            key=lambda r: r["objective"])),
        ("oracle-best-eheft-alpha", lambda rs: min(
            (r for r in rs if r["algorithm"].startswith("E-EHEFT")),
            key=lambda r: r["objective"])),
    ):
        for n in sorted({k[0] for k in by_n_seed}):
            gaps = [min(100.0 * (pick(by_n_seed[(n, s)])["objective"]
                                 / by_n_seed[(n, s)][0]["opt_objective"] - 1.0), 1e9)
                    for s in sorted({k[1] for k in by_n_seed if k[0] == n})]
            out.append({
                "n": n, "algorithm": tag, "instances": len(gaps),
                "mean_gap_pct": round(statistics.mean(gaps), 3),
                "median_gap_pct": round(statistics.median(gaps), 3),
                "p90_gap_pct": round(pct(gaps, 0.9), 3),
                "max_gap_pct": round(max(gaps), 3),
                "mean_ms_ratio": "", "mean_en_ratio": "",
                "on_front_rate": round(statistics.mean(g <= 0.05 for g in gaps), 3),
                "dominated_rate": "", "worst_seed": "",
            })
    return out


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"已写入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟: n=8, 3 个 seed")
    args = ap.parse_args()

    sizes = {8: 3} if args.quick else SIZES
    total_points = sum(3 ** n * k for n, k in sizes.items())
    print(f"最优性差距实验: 规模 {list(sizes)}, 共枚举 {total_points} 个决策")
    if not HAS_DQN:
        print(f"  注: 跳过 DQN ({DQN_SKIP_REASON})")

    t0 = time.perf_counter()
    gap_rows, front_rows, case = [], [], None
    for n, k in sizes.items():
        rows, fronts, c = run_size(n, range(k))
        gap_rows += rows
        front_rows += fronts
        if c and (case is None or c["heft_gap"] > case["heft_gap"]):
            case = c
    print(f"耗时 {time.perf_counter() - t0:.1f}s, {len(gap_rows)} 条算法记录")

    exp = Path(__file__).resolve().parent
    write_csv(exp / "optimality_gap.csv", GAP_COLS, gap_rows)
    write_csv(exp / "pareto_front.csv", FRONT_COLS, front_rows)

    summary = summarize(gap_rows)
    write_csv(exp / "optimality_summary.csv", SUMMARY_COLS, summary)

    # 案例: 最差实例的逐方案明细 + 前沿
    case_rows = []
    opt = case["opt"]
    case_rows.append({"n": case["n"], "seed": case["seed"], "algorithm": "EXACT-OPT",
                      "makespan": opt["makespan"], "energy": opt["energy"],
                      "objective": opt["objective"], "gap_pct": 0.0,
                      "n_end": opt["per_node"]["end"]})
    for r in sorted(case["rows"], key=lambda r: r["objective"]):
        case_rows.append({"n": r["n"], "seed": r["seed"], "algorithm": r["algorithm"],
                          "makespan": r["makespan"], "energy": r["energy"],
                          "objective": r["objective"], "gap_pct": r["gap_pct"],
                          "n_end": r["n_end"]})
    write_csv(exp / "optimality_gap_case.csv",
              ["n", "seed", "algorithm", "makespan", "energy", "objective",
               "gap_pct", "n_end"], case_rows)

    print("\n按规模聚合 (gap% = 加权目标高于精确最优的百分比):")
    print(f"{'n':>3} {'算法':<24}{'均值':>7}{'中位':>7}{'P90':>7}{'最大':>7}"
          f"{'落前沿':>7}{'被支配':>7}")
    for r in summary:
        if r["algorithm"].startswith("oracle"):
            continue
        print(f"{r['n']:>3} {r['algorithm']:<24}{r['mean_gap_pct']:>7.2f}"
              f"{r['median_gap_pct']:>7.2f}{r['p90_gap_pct']:>7.2f}"
              f"{r['max_gap_pct']:>7.2f}{r['on_front_rate']:>7.2f}"
              f"{r['dominated_rate']:>7.2f}")
    print("\n事后最优参照:")
    for r in summary:
        if r["algorithm"].startswith("oracle"):
            print(f"  n={r['n']:<2} {r['algorithm']:<24} 均值 {r['mean_gap_pct']:>5.2f}%"
                  f"  最大 {r['max_gap_pct']:>6.2f}%")
    print(f"\n案例研究: n={case['n']} seed={case['seed']} "
          f"(HEFT 差距 {case['heft_gap']:.1f}%)")


if __name__ == "__main__":
    main()
