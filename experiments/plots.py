# -*- coding: utf-8 -*-
"""
图表生成脚本: 从 experiments/results.csv 与 alpha_sweep.csv 一键生成报告全部图表
输出到 figures/ 目录 (PNG, 200dpi, 可直接插入 Word/PPT)

用法: python experiments/plots.py
"""

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False

# 统一配色: 传统算法蓝系, 自设计绿, RL 红, 全卸载参照灰
COLORS = {"all-end": "#999999", "all-edge": "#777777", "all-cloud": "#555555",
          "A-random": "#1f77b4", "B-greedy": "#2ca02c", "C-HEFT": "#1f77b4",
          "E-EHEFT-a0.75": "#17becf", "E-EHEFT-a0.5": "#2ca02c",
          "E-EHEFT-a0.25": "#17becf", "D-DQN": "#d62728"}


def load_results():
    rows = []
    with open(ROOT / "experiments" / "results.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n"] = int(r["n"]); r["makespan"] = float(r["makespan"])
        r["energy"] = float(r["energy"]); r["decision_ms"] = float(r["decision_ms"])
    return rows


def agg(rows, key_funcs):
    """按 key_funcs 分组, 返回 {key: (均值, 标准差)}"""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(kf(r) for kf in key_funcs)].append(r)
    return {k: (statistics.mean(x["makespan"] for x in v),
                statistics.stdev(x["makespan"] for x in v),
                statistics.mean(x["energy"] for x in v),
                statistics.stdev(x["energy"] for x in v))
            for k, v in groups.items()}


def grouped_bars(rows, metric, title, fname, bw="normal"):
    """图1/2: 算法 × 规模 的时延/能耗分组柱状图"""
    idx = {"makespan": (0, 1), "energy": (2, 3)}[metric]
    algos = ["all-end", "all-edge", "all-cloud", "A-random", "B-greedy",
             "C-HEFT", "E-EHEFT-a0.5", "D-DQN"]
    labels = ["全在端", "全在边", "全在云", "A随机", "B贪心",
              "C-HEFT", "E-EHEFT(α=0.5)", "D-DQN"]
    data = agg([r for r in rows if r["bw"] == bw],
               [lambda r: r["algorithm"], lambda r: r["n"]])
    ns = sorted({r["n"] for r in rows if r["bw"] == bw})
    width = 0.8 / len(ns)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, n in enumerate(ns):
        xs = [i + j * width for i in range(len(algos))]
        means = [data[(a, n)][idx[0]] for a in algos]
        stds = [data[(a, n)][idx[1]] for a in algos]
        ax.bar(xs, means, width * 0.9, yerr=stds, capsize=2,
               label=f"n={n}", color=plt.cm.viridis(0.25 + 0.25 * j))
    ax.set_xticks([i + width for i in range(len(algos))])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("总时延 (s)" if metric == "makespan" else "终端能耗 (J)")
    ax.set_title(f"{title} (正常带宽, 10 seeds 均值±标准差)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=200)
    plt.close(fig)
    print(f"已生成 {fname}")


def pareto(rows):
    """图3: 时间-能耗 Pareto 散点 + E-HEFT α 细扫曲线"""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # 主数据: 各算法在 normal/n=50 下的均值
    data = agg([r for r in rows if r["bw"] == "normal" and r["n"] == 50],
               [lambda r: r["algorithm"]])
    labels = {"all-end": "全在端", "all-edge": "全在边", "all-cloud": "全在云",
              "A-random": "A随机", "B-greedy": "B贪心", "C-HEFT": "C-HEFT",
              "E-EHEFT-a0.5": "E-EHEFT(α=0.5)", "D-DQN": "D-DQN"}
    # 密集区手动错开标签
    offsets = {"C-HEFT": (7, 7), "B-greedy": (-12, 12), "E-EHEFT-a0.5": (8, -15),
               "all-cloud": (7, -13), "D-DQN": (7, 4)}
    for a, lab in labels.items():
        m, _, e, _ = data[(a,)]
        ax.scatter(m, e, s=90, color=COLORS[a], zorder=3,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(lab, (m, e), textcoords="offset points",
                    xytext=offsets.get(a, (7, 4)), fontsize=9)
    # E-HEFT α 细扫曲线
    sweep_csv = ROOT / "experiments" / "alpha_sweep.csv"
    if sweep_csv.exists():
        with open(sweep_csv, encoding="utf-8") as f:
            sw = list(csv.DictReader(f))
        ax.plot([float(r["makespan"]) for r in sw],
                [float(r["energy"]) for r in sw], "o-", color="#2ca02c",
                alpha=0.6, markersize=4, label="E-HEFT α 0→1 细扫")
        ax.legend()
    ax.set_xlabel("平均总时延 (s)")
    ax.set_ylabel("平均终端能耗 (J)")
    ax.set_title("时间-能耗权衡 (n=50, 正常带宽)")
    ax.grid(alpha=0.3)
    # 左下角密集区局部放大
    axins = ax.inset_axes([0.55, 0.08, 0.42, 0.42])
    axins.set_xlim(3.5, 13)
    axins.set_ylim(0, 18)
    for a, lab in labels.items():
        m, _, e, _ = data[(a,)]
        axins.scatter(m, e, s=70, color=COLORS[a], zorder=3,
                      edgecolors="black", linewidths=0.5)
        axins.annotate(lab, (m, e), textcoords="offset points",
                       xytext=offsets.get(a, (7, 4)), fontsize=8)
    if sweep_csv.exists():
        axins.plot([float(r["makespan"]) for r in sw],
                   [float(r["energy"]) for r in sw], "o-", color="#2ca02c",
                   alpha=0.6, markersize=3)
    ax.indicate_inset_zoom(axins, edgecolor="gray")
    axins.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_pareto.png", dpi=200)
    plt.close(fig)
    print("已生成 fig3_pareto.png")


def dqn_convergence():
    """图4: DQN 训练收敛曲线(解析训练日志)"""
    log = ROOT / "experiments" / "dqn_seed3.log"
    if not log.exists():
        print("跳过 fig4 (无训练日志)")
        return
    eps, objs, hefts = [], [], []
    for line in open(log, encoding="utf-8"):
        m = re.search(r"episode\s+(\d+)/\d+\s+ε=[\d.]+\s+DQN加权目标=([\d.]+)\s+\(HEFT=([\d.]+)", line)
        if m:
            eps.append(int(m.group(1)))
            objs.append(float(m.group(2)))
            hefts.append(float(m.group(3)))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(eps, objs, "o-", color="#d62728", label="DQN 加权目标")
    ax.axhline(hefts[-1] if hefts else 0, color="#1f77b4", linestyle="--",
               label="HEFT 参照")
    ax.set_xlabel("训练 episode 数")
    ax.set_ylabel("加权目标 (越小越好)")
    ax.set_title("DQN 训练收敛过程 (评估集, λ=0.5)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_dqn_convergence.png", dpi=200)
    plt.close(fig)
    print("已生成 fig4_dqn_convergence.png")


def dqn_generalization(rows):
    """图5: DQN 与 HEFT 跨规模对比(泛化性)"""
    data = agg([r for r in rows if r["bw"] == "normal"
                and r["algorithm"] in ("C-HEFT", "D-DQN")],
               [lambda r: r["algorithm"], lambda r: r["n"]])
    ns = sorted({r["n"] for r in rows if r["bw"] == "normal"})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, color, lab in [("C-HEFT", "#1f77b4", "C-HEFT"),
                             ("D-DQN", "#d62728", "D-DQN")]:
        ms = [data[(algo, n)][0] for n in ns]
        ax.plot(ns, ms, "o-", color=color, label=lab)
        for n, m in zip(ns, ms):
            ax.annotate(f"{m:.1f}", (n, m), textcoords="offset points",
                        xytext=(0, 7), fontsize=8, ha="center")
    ax.set_xticks(ns)
    ax.set_xlabel("DAG 规模 (子任务数)")
    ax.set_ylabel("平均总时延 (s)")
    ax.set_title("DQN 泛化性: 训练于 n≈10-14, 在更大图上的表现 (正常带宽)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_dqn_generalization.png", dpi=200)
    plt.close(fig)
    print("已生成 fig5_dqn_generalization.png")


def decision_time(rows):
    """图6: 各算法决策耗时(对数轴)"""
    data = agg([r for r in rows if r["bw"] == "normal" and r["n"] == 100],
               [lambda r: r["algorithm"]])
    algos = ["A-random", "B-greedy", "C-HEFT", "E-EHEFT-a0.5", "D-DQN"]
    labels = ["A随机", "B贪心", "C-HEFT", "E-EHEFT", "D-DQN"]
    groups = defaultdict(list)
    for r in rows:
        if r["bw"] == "normal" and r["n"] == 100:
            groups[r["algorithm"]].append(r["decision_ms"])
    ms = [statistics.mean(groups[a]) for a in algos]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, ms, color=[COLORS[a] for a in algos])
    ax.set_yscale("log")
    ax.set_ylabel("单次决策耗时 (ms, 对数轴)")
    ax.set_title("算法决策开销对比 (n=100)")
    for b, v in zip(bars, ms):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_decision_time.png", dpi=200)
    plt.close(fig)
    print("已生成 fig6_decision_time.png")


if __name__ == "__main__":
    rows = load_results()
    grouped_bars(rows, "makespan", "各算法总时延对比", "fig1_makespan.png")
    grouped_bars(rows, "energy", "各算法终端能耗对比", "fig2_energy.png")
    pareto(rows)
    dqn_convergence()
    dqn_generalization(rows)
    decision_time(rows)
    print(f"\n全部图表输出目录: {FIG_DIR}")
