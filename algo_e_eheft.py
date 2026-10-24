# -*- coding: utf-8 -*-
"""
算法E: E-HEFT 能耗感知 HEFT（本组自设计算法，课题要求1 的"设计"部分）

参考: HEFT (Topcuoglu et al., IEEE TPDS 2002) 的列表调度框架，
      将选层准则由"最早完成时间 EFT"改为"归一化时间-能耗加权和"。

设计思想:
  HEFT 只优化 makespan，在本课题的代价模型下终端能耗很高（大量数据经终端
  收发）。E-HEFT 保持 HEFT 的 rank 优先级排序不变，把每个候选层 k 的代价改为

      cost(i,k) = α · EFT(i,k)/T_ref + (1-α) · ΔE(i,k)/E_ref

  其中 EFT 含输入上传/前驱传输/同层排队/结果回传（与仿真器时序模型一致）；
  ΔE 为该决策带来的终端能耗增量:
      在端执行: 0.8·t_exec + 1.2·(经终端的传输时间)
      在边/云:  0.1·t_exec(端空闲) + 1.2·(经终端的传输时间)
  T_ref = Σw/f_end、E_ref = 0.8·Σw/f_end 为"全在端执行"的参考量纲，
  消除时间(秒)与能耗(焦耳)的单位差异。

适用条件: 需要在时延与能耗之间权衡的场景; α 是使用方按业务偏好选择的旋钮。
实现流程: 计算 rank(同 HEFT) → 按优先级拓扑序遍历 → 逐任务对 3 层计算
          EFT 与能耗增量 → 取加权代价最小者。
复杂度: 时间 O(V^2 · P), P=3(层数), 与 HEFT 同阶; 空间 O(V+E)。
重要性质: α=1 时与标准 HEFT 完全等价（归一化不改变 argmin），可用作正确性检验。
负责人: (填组员名字)
"""

from dag_simulator import NODES, topo_order, rank_upward
from algo_c_heft import _transfer, _ready_time, LOCS


def eheft(dag, alpha=0.5):
    """输入 dag 与权衡系数 alpha∈[0,1] (1=纯时延, 0=纯节能)，返回卸载决策"""
    # 参考量纲: 全在端执行的总时间与总能耗
    t_ref = sum(dag["workload"].values()) / NODES["end"]["f"]
    e_ref = NODES["end"]["p_exec"] * t_ref

    finish = {}          # 已定任务的实际完成时刻(与 simulate 时序模型一致)
    node_free = {k: 0.0 for k in LOCS}
    decision = {}

    def _cost(i, k):
        """返回 (加权代价, start, exec_fin, done, e_incr)"""
        start = max(_ready_time(dag, i, k, decision, finish), node_free[k])
        t_exec = dag["workload"][i] / NODES[k]["f"]
        eft = start + t_exec
        # 终端能耗增量
        e = t_exec * (NODES["end"]["p_exec"] if k == "end"
                      else NODES["end"]["p_idle"])
        d_in = dag.get("input_data", {}).get(i, 0.0)
        if d_in > 0 and k != "end":
            e += P_TX * _transfer("end", k, d_in)
        for p in dag["pred"].get(i, []):
            if "end" in (decision.get(p), k):
                e += P_TX * _transfer(decision.get(p), k, dag["data_size"][(p, i)])
        done = eft
        d_out = dag.get("output_data", {}).get(i, 0.0)
        if d_out > 0 and k != "end":
            tt = _transfer(k, "end", d_out)
            done += tt
            e += P_TX * tt
        # 时间项用含回传的 done, 与 HEFT 的 EFT 定义一致 → alpha=1 时严格等价
        cost = alpha * (done / t_ref) + (1 - alpha) * (e / e_ref)
        return cost, start, eft, done, e

    for i in topo_order(dag, priority=rank_upward(dag)):
        best = min(LOCS, key=lambda k: _cost(i, k)[0])
        _, start, exec_fin, done, _ = _cost(i, best)
        decision[i] = best
        finish[i] = done
        node_free[best] = exec_fin
    return decision


P_TX = 1.2  # 与 dag_simulator.P_TX_END 保持一致


if __name__ == "__main__":
    from dag_simulator import gen_dag, simulate
    dag = gen_dag(n=30, seed=42)
    print(f"[算法E E-HEFT] 30任务 seed=42, α 扫描:")
    for a in (1.0, 0.75, 0.5, 0.25, 0.0):
        r = simulate(dag, eheft(dag, alpha=a))
        print(f"  α={a:<5} makespan={r['makespan']:>8.3f}s  能耗={r['energy']:>7.3f}J")
