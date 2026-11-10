"""图 11：与精确最优解的差距（optimality gap）。

左图：n=10 时各算法加权目标相对精确最优的平均/中位差距（越小越好）。
右图：差距最大的实例（n=10, seed=9）上，真实 Pareto 前沿与各算法落点。

数据来自 optimality_gap.py 生成的三个 CSV，不依赖 torch/matplotlib 之外的东西。
用法：
    python experiments/plot_optimality.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(ROOT, "figures")
sys.path.insert(0, ROOT)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 与 plots.py 的配色保持一致（同一算法在不同图里颜色相同）
COLORS = {
    "all-end": "#B0B7C3",
    "all-edge": "#7F8C9B",
    "all-cloud": "#55606E",
    "A-random": "#9E9E9E",
    "B-greedy": "#8D6E63",
    "C-HEFT": "#E4572E",
    "E-EHEFT-a0.25": "#F2A541",
    "E-EHEFT-a0.5": "#F6C453",
    "E-EHEFT-a0.75": "#D98E04",
    "E-EHEFT-a1.0": "#B26A00",
    "D-DQN": "#5B4B8A",
}
LABELS = {
    "all-end": "全在端",
    "all-edge": "全在边",
    "all-cloud": "全在云",
    "A-random": "A 随机",
    "B-greedy": "B 贪心",
    "C-HEFT": "C HEFT",
    "E-EHEFT-a0.25": "E-HEFT α=0.25",
    "E-EHEFT-a0.5": "E-HEFT α=0.5",
    "E-EHEFT-a0.75": "E-HEFT α=0.75",
    "E-EHEFT-a1.0": "E-HEFT α=1.0",
    "D-DQN": "D DQN",
    "oracle-best-single": "参照·逐实例选最优算法",
    "oracle-best-eheft-alpha": "参照·逐实例选最优 α",
}
ORACLE = ("oracle-best-single", "oracle-best-eheft-alpha")


def read_csv(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        sys.exit(f"缺少 {name}，请先运行 experiments/optimality_gap.py")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def left_panel(ax, summary, n=10):
    rows = [r for r in summary if int(r["n"]) == n and r["instances"]]
    rows.sort(key=lambda r: float(r["mean_gap_pct"]))
    names = [r["algorithm"] for r in rows]
    means = [float(r["mean_gap_pct"]) for r in rows]
    medians = [float(r["median_gap_pct"]) for r in rows]
    ypos = range(len(rows))

    ax.barh(list(ypos), means,
            color=[COLORS.get(a, "#999999") if a not in ORACLE else "#CFD8DC"
                   for a in names],
            hatch=["//" if a in ORACLE else None for a in names],
            edgecolor="white", linewidth=0.8, height=0.72, zorder=3)
    # 白色竖线标出中位数：均值被少数灾难实例拉高时，中位数更能代表典型表现
    ax.scatter(medians, list(ypos), marker="|", s=160, color="white",
               linewidths=2.0, zorder=4)
    for y, m, med in zip(ypos, means, medians):
        if m > 100:  # 超长条形把数值写在条内，避免贴边被裁掉
            ax.text(m - 2, y, f"{m:.1f}%", va="center", ha="right",
                    fontsize=8.5, color="white")
        else:
            ax.text(m + 1.2, y, f"{m:.1f}%", va="center", fontsize=8.5, color="#333333")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels([LABELS.get(a, a) for a in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("加权目标高于精确最优的百分比（%，越小越好）", fontsize=10)
    ax.set_title(f"(a) n={n}：与精确最优的差距（条形=均值，白线=中位数）",
                 fontsize=11, loc="left")
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.set_xlim(0, max(means) * 1.18)


def right_panel(ax, gap_rows, front_rows, n=10, seed=9):
    front = sorted([r for r in front_rows
                    if int(r["n"]) == n and int(r["seed"]) == seed],
                   key=lambda r: float(r["makespan"]))
    if not front:
        sys.exit(f"pareto_front.csv 中没有 n={n}, seed={seed} 的前沿数据")
    fx = [float(r["makespan"]) for r in front]
    fy = [float(r["energy"]) for r in front]

    rows = [r for r in gap_rows if int(r["n"]) == n and int(r["seed"]) == seed]
    # 同一落点只画一次（DQN 常与 all-edge 重合）
    seen = set()
    for r in rows:
        key = (round(float(r["makespan"]), 3), round(float(r["energy"]), 3))
        if key in seen:
            continue
        seen.add(key)
        alg = r["algorithm"]
        if alg.startswith("E-EHEFT-a") and alg not in ("E-EHEFT-a0.5", "E-EHEFT-a0.75"):
            continue
        ax.scatter(float(r["makespan"]), float(r["energy"]), s=46,
                   color=COLORS.get(alg, "#999999"), edgecolor="white",
                   linewidth=0.7, zorder=4)
        if alg in ("C-HEFT", "all-edge", "E-EHEFT-a0.5"):
            offset = {"C-HEFT": (-12, 12), "E-EHEFT-a0.5": (8, 8),
                      "all-edge": (8, 7)}[alg]
            ha = "right" if alg == "C-HEFT" else "left"
            ax.annotate(f"{LABELS.get(alg, alg)}\n+{float(r['gap_pct']):.0f}%",
                        (float(r["makespan"]), float(r["energy"])),
                        textcoords="offset points", xytext=offset,
                        ha=ha, fontsize=8, color="#333333")

    ax.plot(fx, fy, "-o", color="#2E86AB", markersize=4.5, linewidth=1.4,
            label="真实 Pareto 前沿", zorder=5)
    ax.scatter([fx[0]], [fy[0]], marker="*", s=230, color="#E4572E",
               edgecolor="white", linewidth=0.8, zorder=6, label="精确最优（λ=0.5）")
    ax.annotate("精确最优", (fx[0], fy[0]), textcoords="offset points",
                xytext=(10, 10), fontsize=9, color="#E4572E")

    ax.set_xlabel("完成时间 makespan（s）", fontsize=10)
    ax.set_ylabel("手机能耗（J）", fontsize=10)
    ax.set_title(f"(b) 最难实例 n={n}, seed={seed}：算法落点 vs 真实前沿",
                 fontsize=11, loc="left")
    ax.grid(linestyle=":", alpha=0.5, zorder=0)
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[a],
                      label=LABELS.get(a, a)) for a in ("C-HEFT", "all-edge", "D-DQN")]
    handles += [Line2D([], [], color="#2E86AB", marker="o", markersize=4,
                       label="真实 Pareto 前沿")]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.92)


def main():
    summary = read_csv("optimality_summary.csv")
    gaps = read_csv("optimality_gap.csv")
    front = read_csv("pareto_front.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    left_panel(axes[0], summary, n=10)
    right_panel(axes[1], gaps, front, n=10, seed=9)
    fig.suptitle("精确最优基准：启发式与 DQN 离理论最优还有多远", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig11_optimality_gap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[ok] 已生成 {out}")


if __name__ == "__main__":
    main()
