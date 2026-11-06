#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键复现全部实验数据与图表（跨平台版，Windows 无 bash 时用这个）

用法:
    python run_all.py                # 校准 → 矩阵 → α 扫 → 敏感性 → 形状 → 出图
    python run_all.py --skip-plots   # 不生成图表(没装 matplotlib 时)

与 run_all.sh 等价；区别是用 sys.executable 调用各步骤，Windows 下也能跑。
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("仿真器手算校准", ["verify_handcheck.py"]),
    ("全量实验矩阵 (600 条)", ["experiments/run_matrix.py"]),
    ("E-HEFT alpha 细扫", ["experiments/alpha_sweep.py"]),
    ("敏感性分析 (带宽/lambda)", ["experiments/sensitivity.py"]),
    ("DAG 形状实验", ["experiments/shape_exp.py"]),
    ("生成全部图表", ["experiments/plots.py"]),
]


def main():
    parser = argparse.ArgumentParser(description="一键复现实验数据与图表")
    parser.add_argument("--skip-plots", action="store_true", help="跳过绘图步骤")
    args = parser.parse_args()

    steps = STEPS[:-1] if args.skip_plots else STEPS
    for idx, (name, cmd) in enumerate(steps, 1):
        print(f"\n========== [{idx}/{len(steps)}] {name} ==========", flush=True)
        result = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        if result.returncode != 0:
            print(f"步骤失败: {name}", file=sys.stderr)
            return result.returncode

    print("\n全部完成: 数据在 experiments/*.csv, 图表在 figures/。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
