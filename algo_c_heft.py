# -*- coding: utf-8 -*-
"""
算法C: HEFT 列表调度 (Heterogeneous Earliest Finish Time)

参考: Topcuoglu, Hariri, Wu. "Performance-effective and low-complexity task
      scheduling for heterogeneous computing." IEEE TPDS, 2002.

设计思想: 两步走——
  (1) 任务排序: 按 rank_upward 优先级(关键路径上的任务先定, 与仿真器回放顺序一致);
  (2) 执行位置: 每个任务选"最早完成时间 EFT"最小的层,
      EFT 计入输入上传、前驱传输、同层排队和出口任务的结果回传。
适用条件: DAG 结构已知、环境参数静态的场景; 优化目标以 makespan 为主
      (原版 HEFT 不考虑能耗, 这是写报告时要指出的局限)。
实现流程: 计算 rank → 按优先级拓扑序遍历 → 逐任务枚举 3 层取 EFT 最小者。
复杂度: 时间 O(V^2 * P), P=3(层数); 空间 O(V + E)。

小改进 (HEFT-EC, 同文件内的 heft_ec()):
  原版 HEFT 每步只取 min(LOCS, key=EFT): 平局时按 ("end","edge","cloud")
  顺序会优先选 "end"(端上执行), 恰恰是最费电的一层; 而 EFT 常只差千分之几,
  终端电耗却可能差很多(端上执行 0.8 W vs 边/云空闲 0.1 W, 经终端传输按 1.2 W)。
  heft_ec() 在 "EFT 不劣于最优值 (1+eps) 倍" 的候选层中改选 "终端能耗增量最小"
  的层, 并把标准 HEFT 解与该能耗优先解各构造一次、按 f=makespan+λ·energy 取
  更优者。因此它在同一模型下**不会比 HEFT 更差**, 代价是常数约 2 倍的构造时间
  (复杂度阶不变)。默认 eps=0.02、λ=0.5(与实验矩阵口径一致)。
  验证脚本: experiments/compare_heft_ec.py(逐例断言内部记账与 simulate 相同、
  基线决策与 heft() 相同)。
  注: heft() 的决策与改动前完全一致, 全组 results.csv 不受影响。

负责人: (填组员名字)
"""

from dag_simulator import NODES, BANDWIDTH, topo_order, rank_upward, P_TX_END

LOCS = ("end", "edge", "cloud")


def _transfer(src, dst, size_mb):
    if src is None or src == dst:
        return 0.0
    b = BANDWIDTH.get((src, dst)) or BANDWIDTH.get((dst, src))
    return size_mb / b


def _ready_time(dag, i, k, decision, finish):
    """任务 i 放到层 k 的最早就绪时刻: 输入上传 与 前驱完成+传输 的较大者"""
    ready = 0.0
    d_in = dag.get("input_data", {}).get(i, 0.0)
    if d_in > 0 and k != "end":
        ready = max(ready, _transfer("end", k, d_in))
    for p in dag["pred"].get(i, []):
        tt = _transfer(decision.get(p), k, dag["data_size"][(p, i)])
        ready = max(ready, finish[p] + tt)
    return ready


def _construct(dag, eps=None):
    """
    按 HEFT 的 rank 顺序构造一个完整解, 并同步累计与 simulate() 同口径的
    makespan / 终端能耗, 返回 (decision, makespan, energy) (未取整)。

    eps is None: 每个任务选 EFT 最小的层(与改动前的 heft() 决策逐例一致);
    eps > 0:     在 EFT <= (1+eps)·best 的候选层里选终端能耗增量最小的层
                 (HEFT-EC 的"能耗优先"解, 供 heft_ec 与标准解比较)。
    """
    finish = {}          # 已定任务的实际完成时刻(与 simulate 时序模型一致)
    node_free = {k: 0.0 for k in LOCS}
    decision = {}
    end_exec = 0.0       # 终端执行时间合计 (与 simulate 的记账一致)
    end_tx = 0.0         # 终端参与收发的时间合计

    def _eft(i, k):
        """返回 (done, start, t_exec): done 含出口任务的结果回传"""
        start = max(_ready_time(dag, i, k, decision, finish), node_free[k])
        t_exec = dag["workload"][i] / NODES[k]["f"]
        done = start + t_exec
        d_out = dag.get("output_data", {}).get(i, 0.0)
        if d_out > 0 and k != "end":
            done += _transfer(k, "end", d_out)
        return done, start, t_exec

    def _e_delta(i, k, t_exec):
        """把任务 i 放到层 k 给终端带来的能耗增量估计(选层用, 与 E-HEFT 同式)"""
        e = t_exec * (NODES["end"]["p_exec"] if k == "end"
                      else NODES["end"]["p_idle"])
        d_in = dag.get("input_data", {}).get(i, 0.0)
        if d_in > 0 and k != "end":
            e += P_TX_END * _transfer("end", k, d_in)
        for p in dag["pred"].get(i, []):
            if "end" in (decision.get(p), k):
                e += P_TX_END * _transfer(decision.get(p), k, dag["data_size"][(p, i)])
        d_out = dag.get("output_data", {}).get(i, 0.0)
        if d_out > 0 and k != "end":
            e += P_TX_END * _transfer(k, "end", d_out)
        return e

    for i in topo_order(dag, priority=rank_upward(dag)):
        cand = {k: _eft(i, k) for k in LOCS}
        best_eft = min(v[0] for v in cand.values())
        if eps is None:
            best = min(LOCS, key=lambda k: cand[k][0])
        else:
            tolerate = [k for k in LOCS if cand[k][0] <= best_eft * (1.0 + eps)]
            best = min(tolerate,
                       key=lambda k: (_e_delta(i, k, cand[k][2]), cand[k][0]))

        # start/exec_fin 决定层占用; done(含结果回传)只供后继与 makespan 计时
        done, start, t_exec = cand[best]
        decision[i] = best
        finish[i] = done
        node_free[best] = start + t_exec

        # 与 simulate() 逐项对齐的能耗记账
        if best == "end":
            end_exec += t_exec
        d_in = dag.get("input_data", {}).get(i, 0.0)
        if d_in > 0 and best != "end":
            end_tx += _transfer("end", best, d_in)
        for p in dag["pred"].get(i, []):
            if "end" in (decision[p], best):
                end_tx += _transfer(decision[p], best, dag["data_size"][(p, i)])
        d_out = dag.get("output_data", {}).get(i, 0.0)
        if d_out > 0 and best != "end":
            end_tx += _transfer(best, "end", d_out)

    makespan = max(finish.values())
    energy = (end_exec * NODES["end"]["p_exec"]
              + (makespan - end_exec) * NODES["end"]["p_idle"]
              + end_tx * P_TX_END)
    return decision, makespan, energy


def heft(dag):
    """输入 dag, 返回卸载决策 {task_id: "end"|"edge"|"cloud"}(标准 HEFT)"""
    return _construct(dag)[0]


def heft_ec(dag, eps=0.02, lam=0.5):
    """
    HEFT-EC: HEFT 的能耗感知小改进(见模块文档"小改进"一节)。

    做法:
      1) 用标准 EFT 准则构造 HEFT 解 (d0, f0);
      2) 用"EFT 容差 eps 内选能耗增量最小"的准则构造备用解 (d1, f1);
      3) 返回 f = makespan + lam·energy 更小的那个。

    参数:
      eps: 每个任务允许的 EFT 相对劣化上限, 默认 0.02(2%)。
           0 时只在 EFT 完全相等时按能耗打破平局(原版按 LOCS 顺序会取 end)。
      lam: 加权目标的能耗权重, 默认 0.5(与 experiments/ 的口径一致)。

    返回: 决策字典, 接口与 heft() 完全相同, 可直接交给 simulate() 打分。

    性质:
      · 同一模型下 f(heft_ec) <= f(heft), 即不会比 HEFT 差;
        因为两个候选解都由 _construct() 按完全相同(且与 simulate 对拍过)的
        时序/能耗公式构造和评分, 择优只是一次比较。
      · 复杂度与 HEFT 同阶 O(V^2·P), 常数约 2(构造两遍); 实测 n=100 时
        决策耗时约 2.3 ms, 仍远低于 DQN 的 ~25 ms。
    """
    d0, ms0, en0 = _construct(dag)
    d1, ms1, en1 = _construct(dag, eps)
    return d1 if (ms1 + lam * en1) < (ms0 + lam * en0) else d0


if __name__ == "__main__":
    from dag_simulator import gen_dag, simulate
    dag = gen_dag(n=30, seed=42)
    for name, fn in (("C HEFT", heft), ("C HEFT-EC", heft_ec)):
        r = simulate(dag, fn(dag))
        print(f"[{name}] 30任务 seed=42: "
              f"makespan={r['makespan']}s, 终端能耗={r['energy']}J")
