# -*- coding: utf-8 -*-
"""
算法B: 贪心卸载 (Greedy Offloading)

设计思想: 按拓扑序逐个任务决策, 每个任务只看自己——选"就绪时间 + 执行时间"
          最小的层。就绪时间包含: 原始输入上传(入口任务)与前驱完成+中间数据传输。
适用条件: 需要快速得到一个"还过得去"的方案时; 任务间耦合弱、传输开销小的场景。
实现流程: 拓扑序遍历 → 对每个任务枚举 3 层 → 取代价最小者。
复杂度: 时间 O(V * P), P=3(层数), 即 O(V); 空间 O(V)。
已知缺点(写报告用): 完全短视——不考虑同层排队拥挤, 也不考虑结果回传和
          对后续任务的影响, 容易把大量任务堆到同一层造成瓶颈。
负责人: (填组员名字)
"""

from dag_simulator import NODES, BANDWIDTH, topo_order

LOCS = ("end", "edge", "cloud")


def _transfer(src, dst, size_mb):
    if src is None or src == dst:
        return 0.0
    b = BANDWIDTH.get((src, dst)) or BANDWIDTH.get((dst, src))
    return size_mb / b


def greedy_offload(dag):
    """输入 dag, 返回卸载决策 {task_id: "end"|"edge"|"cloud"}"""
    finish = {}
    decision = {}
    for i in topo_order(dag):
        best_loc, best_cost = None, float("inf")
        for k in LOCS:
            # 就绪时刻: 输入上传 与 前驱完成+传输 的较大者
            ready = 0.0
            d_in = dag.get("input_data", {}).get(i, 0.0)
            if d_in > 0 and k != "end":
                ready = max(ready, _transfer("end", k, d_in))
            for p in dag["pred"].get(i, []):
                tt = _transfer(decision.get(p), k, dag["data_size"][(p, i)])
                ready = max(ready, finish[p] + tt)
            cost = ready + dag["workload"][i] / NODES[k]["f"]
            if cost < best_cost:
                best_loc, best_cost = k, cost
        decision[i] = best_loc
        finish[i] = best_cost
    return decision


if __name__ == "__main__":
    from dag_simulator import gen_dag, simulate
    dag = gen_dag(n=30, seed=42)
    r = simulate(dag, greedy_offload(dag))
    print(f"[算法B 贪心卸载] 30任务 seed=42: "
          f"makespan={r['makespan']}s, 终端能耗={r['energy']}J")
