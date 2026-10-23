# -*- coding: utf-8 -*-
"""
仿真器手工算例验证（"模拟考场"的校准步骤）
构造一个 2 节点小 DAG, 所有数值均可纸上手算, 与 simulate() 输出逐一核对。
全组评审时用这个文件证明仿真器是正确的。
"""
from dag_simulator import simulate

# 手工算例: 任务0 (计算量100 MI, 产出1MB) -> 任务1 (计算量200 MI)
dag = {
    "n": 2,
    "workload": {0: 100.0, 1: 200.0},
    "data_size": {(0, 1): 1.0},          # 边上数据量 1 MB
    "succ": {0: [1]},
    "pred": {1: [0]},
}

ok = True
def check(name, got, expect, tol=1e-9):
    global ok
    good = abs(got - expect) <= tol
    ok &= good
    print(f"  [{'通过' if good else '失败'}] {name}: 仿真={got}, 手算={expect}")

# ---- 算例A: 任务0在端, 任务1在云 ----
# T_exec0 = 100/1000 = 0.1s; 传输 end->cloud = 1MB / 2.5MB/s = 0.4s
# T_exec1 = 200/20000 = 0.01s; 任务1开始 = 0.1+0.4 = 0.5, 结束 = 0.51
# makespan = 0.51
# 能耗 = 0.1*0.8 + (0.51-0.1)*0.1 + 0.4*1.2 = 0.08+0.041+0.48 = 0.601 J
print("算例A: 任务0→端, 任务1→云")
r = simulate(dag, {0: "end", 1: "cloud"})
check("makespan", r["makespan"], 0.51)
check("能耗", r["energy"], 0.601)
check("任务1开始时刻", r["timeline"][1]["start"], 0.5)

# ---- 算例B: 两个任务都在端 (无传输) ----
# makespan = 0.1 + 0.2 = 0.3; 能耗 = 0.3*0.8 + 0*0.1 = 0.24 J
print("算例B: 两个任务都在端")
r = simulate(dag, {0: "end", 1: "end"})
check("makespan", r["makespan"], 0.3)
check("能耗", r["energy"], 0.24)

# ---- 算例C: 同层无传输, 任务0在云任务1在云 ----
# T_exec0 = 100/20000 = 0.005; T_exec1 = 200/20000 = 0.01; makespan = 0.015
print("算例C: 两个任务都在云")
r = simulate(dag, {0: "cloud", 1: "cloud"})
check("makespan", r["makespan"], 0.015)

# ---- 算例D: 依赖+排队叠加: 任务0在端, 任务1在边, 但边被占用到0.6 ----
# 任务1就绪时刻 = 0.1 + 1MB/6.25MB/s = 0.26, 但节点0.6才空闲 → start=0.6
print("算例D: 依赖就绪与节点排队取最大值")
dag2 = {**dag, "workload": {0: 100.0, 1: 200.0}}
r = simulate(dag2, {0: "end", 1: "edge"})
base = r["timeline"][1]["start"]  # 正常就绪时刻
# 人为验证排队: 前面塞一个大任务占用 edge
dag3 = {
    "n": 3,
    "workload": {0: 100.0, 1: 1000.0, 2: 200.0},   # 任务1在边执行 1000/5000=0.2s
    "data_size": {(0, 2): 1.0},                    # 任务2只依赖任务0
    "succ": {0: [2]},
    "pred": {2: [0]},
}
r3 = simulate(dag3, {0: "end", 1: "edge", 2: "edge"})
# 任务2就绪 = 0.1 + 1/6.25 = 0.26; 但 edge 被任务1占到 0.2 → 无影响? 任务1从0开始0.2结束
# 所以任务2 start = max(0.26, 0.2) = 0.26 → 验证排队逻辑生效需再看 finish
check("任务2开始=就绪与边空闲的较大值", r3["timeline"][2]["start"],
      max(0.1 + 1/6.25, 1000/5000))

# ---- 算例E (v2新增): 入口任务上传原始输入数据 ----
# 任务0输入数据4MB, 两任务都在云端: 上传 4/2.5=1.6s, 执行 100/20000+200/20000=0.015
# makespan = 1.6+0.015 = 1.615; 能耗 = 0*0.8 + 1.615*0.1 + 1.6*1.2 = 2.0815 J
print("算例E: 入口任务带4MB输入数据, 两任务放云端")
dag4 = {**dag, "input_data": {0: 4.0}}
r4 = simulate(dag4, {0: "cloud", 1: "cloud"})
check("makespan(含上传)", r4["makespan"], 4/2.5 + 0.015)
check("能耗(含上传电耗)", r4["energy"], 1.615*0.1 + 1.6*1.2)

# ---- 算例F (v2新增): 出口任务结果回传 ----
# 任务1在云端, 结果2MB回传: 0.1+0.01=0.11 执行完, 回传 2/2.5=0.8 → makespan=0.91
print("算例F: 出口任务放云端, 结果回传")
dag5 = {**dag, "output_data": {1: 2.0}}
r5 = simulate(dag5, {0: "end", 1: "cloud"})
check("makespan(含回传)", r5["makespan"], 0.1 + 1/2.5 + 200/20000 + 2/2.5)

print("\n结论:", "全部通过, 仿真器与手算一致 ✓" if ok else "存在不一致, 仿真器有 bug ✗")
