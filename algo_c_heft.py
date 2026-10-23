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
负责人: (填组员名字)
"""

from dag_simulator import NODES, BANDWIDTH, topo_order, rank_upward

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


def heft(dag):
    """输入 dag, 返回卸载决策 {task_id: "end"|"edge"|"cloud"}"""
    finish = {}          # 已定任务的实际完成时刻(与 simulate 时序模型一致)
    node_free = {k: 0.0 for k in LOCS}
    decision = {}

    def _eft(i, k):
        start = max(_ready_time(dag, i, k, decision, finish), node_free[k])
        done = start + dag["workload"][i] / NODES[k]["f"]
        d_out = dag.get("output_data", {}).get(i, 0.0)
        if d_out > 0 and k != "end":   # 出口任务的结果回传
            done += _transfer(k, "end", d_out)
        return done

    for i in topo_order(dag, priority=rank_upward(dag)):
        best = min(LOCS, key=lambda k: _eft(i, k))
        # start/exec_fin 决定层占用; done(含结果回传)只供后继与 makespan 计时
        start = max(_ready_time(dag, i, best, decision, finish), node_free[best])
        exec_fin = start + dag["workload"][i] / NODES[best]["f"]
        done = exec_fin
        d_out = dag.get("output_data", {}).get(i, 0.0)
        if d_out > 0 and best != "end":
            done += _transfer(best, "end", d_out)
        decision[i] = best
        finish[i] = done
        node_free[best] = exec_fin
    return decision


if __name__ == "__main__":
    from dag_simulator import gen_dag, simulate
    dag = gen_dag(n=30, seed=42)
    r = simulate(dag, heft(dag))
    print(f"[算法C HEFT] 30任务 seed=42: "
          f"makespan={r['makespan']}s, 终端能耗={r['energy']}J")
