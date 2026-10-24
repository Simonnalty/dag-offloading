# 课题10：云边端协同计算的 DAG 任务卸载算法设计

《算法设计与分析》课程课题。在云—边—端三层计算环境下，对 DAG 任务卸载与调度过程建模，
实现并对比多种卸载算法（含 1 种强化学习），分析其性能与复杂度。

## 项目结构

| 文件 | 说明 |
|---|---|
| `dag_simulator.py` | **共享仿真器**（"考场"）：三层环境参数、随机 DAG 生成器、`simulate()` 打分接口 |
| `verify_handcheck.py` | 仿真器校准：6 个可纸上手算的算例，与仿真输出逐一核对（改仿真器后必跑） |
| `algo_a_random.py` | **算法A** 随机卸载 —— baseline 下限 |
| `algo_b_greedy.py` | **算法B** 贪心卸载 —— 每任务独立选代价最小层，短视但快 |
| `algo_c_heft.py` | **算法C** HEFT 列表调度 —— 质量标杆（Topcuoglu et al., 2002） |
| `algo_e_eheft.py` | **算法E** E-HEFT —— 本组自设计：时间-能耗加权选层，α=1 时严格等价于 HEFT，α 扫描产出 Pareto 曲线 |
| `algo_d_dqn.py` | **算法D** DQN 强化学习 —— 环境逐步复刻仿真器时序（已验证一致），MLP+经验回放+目标网络；`dqn_offload(dag)` 加载权重输出决策 |
| `algo_d_dqn_weights.pth` | DQN 训练好的权重（多种子选优后的最优一组） |
| `experiments/train_variant.py` | 多种子训练辅助脚本（`python experiments/train_variant.py <seed>`） |
| `experiments/run_matrix.py` | 批量实验运行器：算法 × 规模 × 带宽场景 × seed 的矩阵，输出 `results.csv` 与汇总表 |
| `experiments/results.csv` | 全量实验数据（`python experiments/run_matrix.py` 重新生成） |

## 环境

Python 3.8+。基础功能无第三方依赖；运行 DQN 相关（`algo_d_dqn.py`、实验矩阵中的 D-DQN）需要：

```bash
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
```

（CPU 版即可：策略网络只有约 5600 参数，GPU 无收益，瓶颈在环境仿真。）

## 快速开始

```bash
# 1. 校准仿真器（任何仿真器改动后都要跑）
python verify_handcheck.py

# 2. 单独跑某个算法
python algo_a_random.py
python algo_b_greedy.py
python algo_c_heft.py
python algo_e_eheft.py        # α 扫描演示

# 3. 批量实验(全量矩阵, 生成 results.csv 与汇总表)
python experiments/run_matrix.py
python experiments/run_matrix.py --quick   # 小矩阵冒烟测试
```

自己写算法只需两步：

```python
from dag_simulator import gen_dag, simulate
dag = gen_dag(n=30, seed=42)
decision = my_algorithm(dag)          # 返回 {task_id: "end"|"edge"|"cloud"}
result = simulate(dag, decision)      # {"makespan": 秒, "energy": 焦耳, ...}
```

## 统一接口约定（勿改，改动需全组评审）

- **决策格式**：`{task_id: "end" | "edge" | "cloud"}` 字典
- **打分入口**：`simulate(dag, decision)`，所有算法共用同一个仿真器
- **任务顺序**：仿真器统一按 `rank_upward` 优先级拓扑序回放，与 HEFT 一致

## 协作方式

1. `main` 分支保持可运行：改动前先拉取，提交前跑一遍 `verify_handcheck.py`
2. 每个算法在自己负责的 `algo_x_*.py` 文件里开发，不要动别人的文件和 `dag_simulator.py`
3. 改 `dag_simulator.py`（公共接口/模型）必须先在群里评审，并补充手算算例
