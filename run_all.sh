#!/usr/bin/env bash
# 一键复现全部实验数据与图表 (答辩演示用)
# 用法: bash run_all.sh    (在仓库根目录执行, 约 15 分钟, 其中 DQN 静态训练约 4 分钟)
set -e
cd "$(dirname "$0")"

echo "========== [1/6] 仿真器手算校准 =========="
python verify_handcheck.py

echo "========== [2/6] 全量实验矩阵 (600 条) =========="
python experiments/run_matrix.py

echo "========== [3/6] E-HEFT alpha 细扫 =========="
python experiments/alpha_sweep.py

echo "========== [4/6] 敏感性分析 (带宽/lambda) =========="
python experiments/sensitivity.py

echo "========== [5/6] DAG 形状实验 =========="
python experiments/shape_exp.py

echo "========== [6/6] 生成全部图表 =========="
python experiments/plots.py

echo ""
echo "全部完成: 数据在 experiments/*.csv, 图表在 figures/。"
echo "注: 动态 DQN 权重(algo_d_dqn_dynamic.pth)训练约 4 分钟/种子, 按需执行:"
echo "    python experiments/train_dynamic.py 11 && python experiments/train_dynamic.py 22"
echo "    python experiments/dynamic_exp.py"
