# -*- coding: utf-8 -*-
"""
精确最优基准: 小规模 DAG 的全空间枚举 (Exact / Brute-force Baseline)

定位: 这不是第六种"卸载算法", 而是给 A/B/C/E 四种启发式打分的**标尺**。
      课题报告此前只有 all-end/all-edge/all-cloud 三个参照, 它们既不是最优解,
      也无法回答"HEFT 离最优到底差多少"。本模块用枚举给出:
        (1) 精确最优解 (给定 λ 下加权目标最小);
        (2) 精确 Pareto 前沿 (makespan 与终端能耗的所有非支配点)。

为什么枚举是精确的:
  仿真器对任意决策 decision 的评分是确定性的; 决策空间恰为 3^V
  (每个任务独立选 端/边/云)。因此遍历全部 3^V 个决策即得到全部可行点,
  再按 (makespan, energy) 做非支配筛选, 结果就是该模型下的真实前沿,
  不依赖任何启发式假设。V=8 时 6561 个点, V=10 时 59049 个点, V=12 时 531441 个点。

规模上限:
  3^V 随 V 指数增长, 本模块只用于小图校准 (V ≤ 12)。大规模结论必须由
  启发式 + 本模块给出的小规模最优性差距外推, 不能直接套用。
  运行时间参考 (单核 Python): V=8 约 0.2 s, V=10 约 2 s, V=12 约 20 s。

与共享仿真器的一致性:
  `evaluate()` 是 `dag_simulator.simulate()` 的等价快速实现(预先算好 rank 回放
  顺序, 去掉 timeline 构造)。`verify_exact.py` 在随机实例上逐点比对两者,
  差异必须为 0 (容差 1e-6)。若日后修改 dag_simulator.py, 必须重跑该脚本。

用法:
    from algo_f_exact import exact_opt, exact_front
    dag = gen_dag(n=10, seed=0)
    best = exact_opt(dag, lam=0.5)      # 加权目标最优决策
    front = exact_front(dag)            # 真实 Pareto 前沿
"""

import itertools

from dag_simulator import (BANDWIDTH, NODES, P_TX_END, rank_upward,
                           topo_order, transfer_time)

LOCS = ("end", "edge", "cloud")
MAX_EXACT_N = 12          # 3^12 = 531441, 再大就不适合纯 Python 枚举
LAM_DEFAULT = 0.5         # 与报告/实验矩阵一致的能耗权重


def objective(makespan, energy, lam=LAM_DEFAULT):
    """加权目标: makespan + λ·energy (报告 5.1 节口径)"""
    return makespan + lam * energy


def _check_size(dag):
    n = dag["n"]
    if n > MAX_EXACT_N:
        raise ValueError(
            f"精确枚举只支持 n≤{MAX_EXACT_N} (3^{n}={3 ** n} 个决策), "
            f"当前 n={n}; 大规模请用启发式并用本模块的小图结论外推")
    return n


def _playback_order(dag):
    """仿真器固定的回放顺序, 枚举时复用, 避免每个决策重算 rank"""
    return topo_order(dag, priority=rank_upward(dag))


def _eval_combo(dag, combo, order):
    """
    给定 combo[tid] ∈ {0,1,2} 计算 (makespan, energy)。

    与 dag_simulator.simulate() 逐行等价:
      start = max(输入上传完成, 前驱完成+传输, 目标层空闲)
      出口任务结果回传计入 makespan 与终端收发能耗
      能耗 = 端执行·P_exec + 端等待·P_idle + 端收发·P_tx
    """
    n = dag["n"]
    finish = [0.0] * n
    node_free = [0.0, 0.0, 0.0]     # 端/边/云 各自空闲时刻
    end_exec = 0.0                  # 终端执行时间合计
    end_tx = 0.0                    # 终端参与收发的时间合计

    workload = dag["workload"]
    pred = dag["pred"]
    data_size = dag["data_size"]
    input_data = dag.get("input_data", {})
    output_data = dag.get("output_data", {})

    for tid in order:
        c = combo[tid]
        loc = LOCS[c]
        ready = 0.0

        d_in = input_data.get(tid, 0.0)
        if d_in > 0.0 and c != 0:                       # 入口数据从终端上传
            tt = transfer_time("end", loc, d_in)
            end_tx += tt
            ready = tt if tt > ready else ready

        for p in pred.get(tid, ()):                     # 依赖约束
            src = LOCS[combo[p]]
            tt = transfer_time(src, loc, data_size[(p, tid)])
            if combo[p] == 0 or c == 0:                 # 任一端经终端收发
                end_tx += tt
            cand = finish[p] + tt
            if cand > ready:
                ready = cand

        t_exec = workload[tid] / NODES[loc]["f"]
        start = ready if ready > node_free[c] else node_free[c]
        exec_fin = start + t_exec
        node_free[c] = exec_fin
        if c == 0:
            end_exec += t_exec

        done = exec_fin
        d_out = output_data.get(tid, 0.0)
        if d_out > 0.0 and c != 0:                      # 出口结果回传终端
            tt = transfer_time(loc, "end", d_out)
            end_tx += tt
            done += tt

        finish[tid] = done

    makespan = max(finish)
    energy = (end_exec * NODES["end"]["p_exec"]
              + (makespan - end_exec) * NODES["end"]["p_idle"]
              + end_tx * P_TX_END)
    return round(makespan, 4), round(energy, 4)


def decision_to_combo(dag, decision):
    """把 {tid: loc} 决策转为枚举用的整数元组, 顺带校验合法性"""
    combo = []
    for i in range(dag["n"]):
        if i not in decision:
            raise ValueError(f"决策缺少任务 {i}")
        loc = decision[i]
        if loc not in LOCS:
            raise ValueError(f"非法位置 {loc!r}")
        combo.append(LOCS.index(loc))
    return tuple(combo)


def evaluate(dag, decision):
    """等价于 simulate() 的快速评分, 返回 (makespan, energy)"""
    return _eval_combo(dag, decision_to_combo(dag, decision), _playback_order(dag))


def _combo_to_decision(combo):
    return {i: LOCS[c] for i, c in enumerate(combo)}


def _per_node(decision):
    return {k: sum(1 for v in decision.values() if v == k) for k in LOCS}


def exact_front(dag, lam=LAM_DEFAULT, verbose=False):
    """
    全空间枚举 + 非支配筛选, 返回真实 Pareto 前沿。

    返回: [{"makespan", "energy", "objective", "decision", "per_node"}, ...]
          按 makespan 升序、energy 严格降序; 任意两点互不支配。
    """
    n = _check_size(dag)
    order = _playback_order(dag)
    points = []
    for combo in itertools.product(range(3), repeat=n):
        ms, en = _eval_combo(dag, combo, order)
        points.append((ms, en, combo))

    # 按 makespan 升序、energy 升序排序后一遍扫描:
    # 只有当 energy 严格低于此前所有点时才进入前沿 → O(P log P), 无需两两比较
    points.sort(key=lambda p: (p[0], p[1]))
    front = []
    best_energy = float("inf")
    for ms, en, combo in points:
        if en < best_energy - 1e-9:
            best_energy = en
            decision = _combo_to_decision(combo)
            front.append({
                "makespan": ms,
                "energy": en,
                "objective": round(objective(ms, en, lam), 4),
                "decision": decision,
                "per_node": _per_node(decision),
            })
    if verbose:
        print(f"n={n}: 枚举 {len(points)} 个决策, 前沿 {len(front)} 个非支配点")
    return front


def exact_opt(dag, lam=LAM_DEFAULT):
    """
    λ 下的精确最优(加权目标最小); 平分时取 makespan 更小者。

    返回 dict: decision / makespan / energy / objective / front_size / points
    """
    front = exact_front(dag, lam=lam)
    best = min(front, key=lambda p: (p["objective"], p["makespan"], p["energy"]))
    n = dag["n"]
    return {
        "decision": best["decision"],
        "makespan": best["makespan"],
        "energy": best["energy"],
        "objective": best["objective"],
        "front_size": len(front),
        "points": 3 ** n,
        "per_node": best["per_node"],
        "lam": lam,
    }


def is_dominated(makespan, energy, front, tol=1e-9):
    """点 (makespan, energy) 是否被前沿中某点严格支配"""
    for p in front:
        if (p["makespan"] <= makespan + tol and p["energy"] <= energy + tol
                and (p["makespan"] < makespan - tol
                     or p["energy"] < energy - tol)):
            return True
    return False


def on_front(makespan, energy, front, tol=5e-4):
    """点是否落在前沿上(与某个前沿点相等, 容差为 4 位小数的半个单位)"""
    return any(abs(p["makespan"] - makespan) <= tol
               and abs(p["energy"] - energy) <= tol for p in front)


if __name__ == "__main__":
    from dag_simulator import gen_dag, simulate
    from algo_c_heft import heft
    from algo_e_eheft import eheft

    dag = gen_dag(n=8, p_edge=0.15, seed=0)
    best = exact_opt(dag, lam=0.5)
    front = exact_front(dag)
    print(f"[精确基准] n={dag['n']} seed=0: 枚举 {best['points']} 个决策, "
          f"前沿 {best['front_size']} 点")
    print(f"  最优加权目标 {best['objective']:.3f} "
          f"(makespan={best['makespan']:.3f}s, energy={best['energy']:.3f}J, "
          f"端/边/云={best['per_node']['end']}/{best['per_node']['edge']}/"
          f"{best['per_node']['cloud']})")
    for name, dec in (("C-HEFT", heft(dag)), ("E-HEFT(0.5)", eheft(dag, 0.5))):
        r = simulate(dag, dec)
        f = objective(r["makespan"], r["energy"])
        print(f"  {name:<11} 加权目标 {f:>6.3f}  差距 {100 * (f / best['objective'] - 1):>6.2f}%")
