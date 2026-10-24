# 小组分工与贡献说明

> 课程提交用。组员姓名/学号请自行填写; "代码佐证"列对应 GitHub 仓库的提交记录, 可直接核对。

| 成员 | 角色 | 负责模块 | 主要文件 | 佐证(提交/产物) |
|---|---|---|---|---|
| 组员1 (组长/架构) | 总架构、接口冻结、进度管理 | 仿真器核心与调度接口 | `dag_simulator.py`, README, 全组评审记录 | 提交记录 + `verify_handcheck.py` 校准 |
| 组员2 | 建模仿真 | 三层环境参数、DAG 生成器、手算校准 | `dag_simulator.py`, `verify_handcheck.py` | 6 个手算算例 |
| 组员3 | 算法A+B | 随机卸载、贪心卸载及说明文字 | `algo_a_random.py`, `algo_b_greedy.py` | 各自文件头的设计文档 |
| 组员4 | 算法C | HEFT 列表调度及复杂度分析 | `algo_c_heft.py`, `docs/complexity.md` | HEFT 实现 + 复杂度文档 |
| 组员5 | 算法E | E-HEFT 自设计算法(α 加权) | `algo_e_eheft.py`, `experiments/alpha_sweep.py` | α=1 等价性验证 + Pareto 数据 |
| 组员6 | 算法D | DQN 设计(状态/动作/奖励)、训练与调参 | `algo_d_dqn.py`, `experiments/train_variant.py`, `train_dynamic.py` | 训练日志、多种子选优记录 |
| 组员7 | 实验 | 实验矩阵、敏感性/形状/动态实验 | `experiments/run_matrix.py`, `sensitivity.py`, `shape_exp.py`, `dynamic_exp.py` | 600+ 条数据、4 份 CSV |
| 组员8 | 文档 | 图表、场景调研、报告与 PPT 整合 | `experiments/plots.py`, `docs/scenarios.md`, `docs/report_draft.md`, figures/ | 10 张图 + 4 份文档 |

## 协作机制(供报告"组织与分工"小节引用)

1. **接口先行**: 第 1 周全组评审冻结两个接口——算法输出格式 `{task_id: 层}` 与仿真器 `simulate(dag, decision)`; 此后各组并行开发互不阻塞。
2. **质量门**: 改公共仿真器必须先跑通 6 个手算算例; main 分支保持可运行。
3. **多种子纪律**: RL 实验一律多种子训练并报告方差(本项目种子间方差达 ±40%, 单种子结论不可信)。
4. **如实验记录**: 负结果如实入报告(DQN 动态环境未超越启发式、混合规模训练泛化恶化), 并给出分析——对应课程目标"适用条件分析"。
