# -*- coding: utf-8 -*-
"""
云边端协同 DAG 任务卸载仿真器（课题10 · 要求1：数学建模与仿真环境） v2

v2 修正: 数据产生在终端设备。入口任务需上传原始输入数据, 出口任务需把结果
回传终端(用户要看到结果)。没有这一项, "全卸载到云"会白捡算力优势, 不真实。

用法:
    from dag_simulator import gen_dag, simulate
    dag = gen_dag(n=10, seed=42)
    decision = {tid: "edge" for tid in dag["workload"]}
    result = simulate(dag, decision)
    print(result["makespan"], result["energy"])

核心接口:
    simulate(dag, decision) -> {"makespan": 秒, "energy": 焦耳(终端), "timeline": [...]}
    decision 形如 {task_id: "end" | "edge" | "cloud"}
"""

import random
from collections import defaultdict

# ---------------- 1. 环境参数（三层节点） ----------------
# f: CPU 速率, 单位 MI/s;  p_exec: 执行功率(W);  p_idle: 空闲功率(W)
NODES = {
    "end":   {"f": 1.0e3, "p_exec": 0.8, "p_idle": 0.1},   # 端: 手机级
    "edge":  {"f": 5.0e3, "p_exec": 10.0, "p_idle": 5.0},  # 边: 边缘服务器
    "cloud": {"f": 20e3, "p_exec": 50.0, "p_idle": 30.0},  # 云: 数据中心
}

# 层间带宽, 单位 MB/s (对称)
BANDWIDTH = {
    ("end", "edge"): 6.25,    # 50 Mbps
    ("end", "cloud"): 2.5,    # 20 Mbps
    ("edge", "cloud"): 12.5,  # 100 Mbps
}

P_TX_END = 1.2  # 终端收发数据的功率 (W), 近似 WiFi

VALID_LOC = ("end", "edge", "cloud")


def transfer_time(src, dst, size_mb):
    """两层之间传 size_mb 兆字节所需时间(秒); 同层传输为 0"""
    if src == dst:
        return 0.0
    b = BANDWIDTH.get((src, dst)) or BANDWIDTH.get((dst, src))
    return size_mb / b


# ---------------- 2. DAG 模型与随机生成器 ----------------
def gen_dag(n=10, p_edge=0.3, seed=0, w_range=(200, 2000), d_range=(0.5, 5.0),
            in_range=(2.0, 8.0), out_range=(0.1, 1.0)):
    """
    随机 DAG: 节点按 0..n-1 拓扑编号, 边只从小指向大(保证无环)。
    workload:  每个任务的计算量 (MI)。默认 200-2000 对应模型推理/视频分析类
               子任务——该标定下"计算-传输权衡"处于有意义区间: 计算量太小
               (如 50-500)时传输开销永远占优, 最优解退化为"全放边缘"。
    data_size: 每条边的传输数据量 (MB), 任务间中间数据
    input_data: 入口任务(无前驱)的原始输入数据 (MB), 产生在终端, 卸载需上传
    output_data: 出口任务(无后继)的结果数据 (MB), 需回传终端给用户
    """
    rng = random.Random(seed)
    workload = {i: round(rng.uniform(*w_range), 1) for i in range(n)}
    data_size = {}
    succ = defaultdict(list)
    pred = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                succ[i].append(j)
                pred[j].append(i)
                data_size[(i, j)] = round(rng.uniform(*d_range), 2)
    for j in range(1, n):
        if not pred[j]:
            i = rng.randrange(j)
            succ[i].append(j)
            pred[j].append(i)
            data_size[(i, j)] = round(rng.uniform(*d_range), 2)
    entries = [i for i in range(n) if not pred[i]]
    exits = [i for i in range(n) if not succ[i]]
    input_data = {i: round(rng.uniform(*in_range), 2) for i in entries}
    output_data = {i: round(rng.uniform(*out_range), 2) for i in exits}
    return {"n": n, "workload": workload, "data_size": data_size,
            "succ": dict(succ), "pred": dict(pred),
            "input_data": input_data, "output_data": output_data}


def rank_upward(dag):
    """
    HEFT 式任务优先级 rank(i) = 平均执行代价 + max(传输代价 + rank(后继))。
    rank 越大的任务越靠近关键路径, 应越早调度。
    同时被仿真器和 HEFT 使用, 保证两边的任务处理顺序一致。
    """
    f_avg = sum(NODES[k]["f"] for k in VALID_LOC) / len(VALID_LOC)
    b_avg = sum(BANDWIDTH.values()) / len(BANDWIDTH)
    rank = {}

    def calc(i):
        if i in rank:
            return rank[i]
        rank[i] = dag["workload"][i] / f_avg + max(
            (dag["data_size"][(i, j)] / b_avg + calc(j)
             for j in dag["succ"].get(i, [])),
            default=0.0,
        )
        return rank[i]

    for i in range(dag["n"]):
        calc(i)
    return rank


def topo_order(dag, priority=None):
    """
    Kahn 拓扑序。priority 提供时, 在"前驱全部完成"的就绪任务中
    优先弹出 priority 大的(堆实现), 这是 HEFT 列表调度的标准做法;
    不提供时按编号顺序(FIFO), 结果确定可复现。
    """
    import heapq
    if priority is None:
        priority = {i: -i for i in range(dag["n"])}   # 等价于按编号 FIFO
    indeg = {i: len(dag["pred"].get(i, [])) for i in range(dag["n"])}
    heap = [(-priority[i], i) for i in range(dag["n"]) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(i)
        for j in dag["succ"].get(i, []):
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, (-priority[j], j))
    assert len(order) == dag["n"], "DAG 存在环!"
    return order


# ---------------- 2b. 分层 DAG 生成器 (宽/深形状实验用) ----------------
def gen_dag_layered(layers, width, seed=0, cross_p=0.15,
                    w_range=(200, 2000), d_range=(0.5, 5.0)):
    """
    分层随机 DAG: 任务分布在 layers 个层级上, 每层 width 个任务;
    边只从第 t 层指向 t+1 层(以 cross_p 概率连边), 保证无环。
    width 大 → 宽扇入形状(工业物联网); layers 大 → 深链式形状(视频分析)。
    接口与 gen_dag 完全一致(input_data/output_data 同样生成)。
    """
    rng = random.Random(seed)
    layer_of = {}
    n = layers * width
    for i in range(n):
        layer_of[i] = i // width
    workload = {i: round(rng.uniform(*w_range), 1) for i in range(n)}
    data_size, succ, pred = {}, defaultdict(list), defaultdict(list)
    for i in range(n):
        for j in range(n):
            if layer_of[j] == layer_of[i] + 1 and rng.random() < cross_p:
                succ[i].append(j); pred[j].append(i)
                data_size[(i, j)] = round(rng.uniform(*d_range), 2)
    for j in range(n):
        if not pred[j] and layer_of[j] > 0:      # 保证除首层外都有前驱
            cands = [i for i in range(n) if layer_of[i] == layer_of[j] - 1]
            i = rng.choice(cands)
            succ[i].append(j); pred[j].append(i)
            data_size[(i, j)] = round(rng.uniform(*d_range), 2)
    entries = [i for i in range(n) if not pred[i]]
    exits = [i for i in range(n) if not succ[i]]
    input_data = {i: round(rng.uniform(2.0, 8.0), 2) for i in entries}
    output_data = {i: round(rng.uniform(0.1, 1.0), 2) for i in exits}
    return {"n": n, "workload": workload, "data_size": data_size,
            "succ": dict(succ), "pred": dict(pred),
            "input_data": input_data, "output_data": output_data}


# ---------------- 2c. 时变带宽仿真 (动态环境) ----------------
def transfer_duration(t0, size_mb, src, dst, phases):
    """
    时变带宽下的传输时长: 速度(t) = B_ab * scale(t)。phases 为分段常数
    [(起始时刻, 带宽缩放), ...](按起始时刻升序, 最后一段延续到无穷)。
    从 t0 起逐段推进, 直到累计传完 size_mb。
    scale 恒为 1 时与静态 transfer_time 结果一致(可断言验证)。
    """
    if src == dst:
        return 0.0
    b = BANDWIDTH.get((src, dst)) or BANDWIDTH.get((dst, src))
    remaining, t = size_mb, t0
    for idx, (t_start, scale) in enumerate(phases):
        if t < t_start:
            t = t_start
        t_end = phases[idx + 1][0] if idx + 1 < len(phases) else float("inf")
        if t >= t_end:
            continue
        speed = b * max(scale, 0.25)         # 带宽下限 25%, 防止除零
        span = t_end - t
        if speed * span >= remaining:
            return (t + remaining / speed) - t0   # 统一返回"时长"(而非绝对时刻)
        remaining -= speed * span
        t = t_end
    return t - t0


def make_telegraph_phases(period, scales, horizon=120.0):
    """分段常数带宽调度: 周期 period, 各段缩放按 scales 循环取值"""
    phases, t, i = [], 0.0, 0
    while t < horizon:
        phases.append((t, scales[i % len(scales)]))
        t += period
        i += 1
    return phases


def simulate_dynamic(dag, decision, phases, nodes=NODES, p_tx=P_TX_END):
    """
    时变带宽仿真: 接口与 simulate() 一致, 仅传输时长改按带宽调度 phases
    逐段积分计算。phases 恒为 scale=1 时结果与 simulate() 相等(可断言)。
    """
    for tid, loc in decision.items():
        assert loc in VALID_LOC, f"非法位置 {loc}"

    finish, node_free = {}, {k: 0.0 for k in VALID_LOC}
    timeline = []
    end_exec = end_tx = 0.0

    for tid in topo_order(dag, priority=rank_upward(dag)):
        loc = decision[tid]
        ready = 0.0
        d_in = dag.get("input_data", {}).get(tid, 0.0)
        if d_in > 0 and loc != "end":
            tt = transfer_duration(0.0, d_in, "end", loc, phases)
            end_tx += tt
            ready = max(ready, tt)
        for p in dag["pred"].get(tid, []):
            tt = transfer_duration(finish[p], dag["data_size"][(p, tid)],
                                   decision[p], loc, phases)
            if "end" in (decision[p], loc):
                end_tx += tt
            ready = max(ready, finish[p] + tt)
        t_exec = dag["workload"][tid] / nodes[loc]["f"]
        start = max(ready, node_free[loc])
        exec_fin = start + t_exec
        node_free[loc] = exec_fin
        if loc == "end":
            end_exec += t_exec
        done = exec_fin
        d_out = dag.get("output_data", {}).get(tid, 0.0)
        if d_out > 0 and loc != "end":
            tt = transfer_duration(exec_fin, d_out, loc, "end", phases)
            end_tx += tt
            done += tt
        finish[tid] = done
        timeline.append({"task": tid, "loc": loc, "start": round(start, 4),
                         "finish": round(exec_fin, 4)})

    makespan = max(finish.values())
    energy = (end_exec * nodes["end"]["p_exec"]
              + (makespan - end_exec) * nodes["end"]["p_idle"]
              + end_tx * p_tx)
    per_node = {k: sum(1 for t in timeline if t["loc"] == k) for k in VALID_LOC}
    return {"makespan": round(makespan, 4), "energy": round(energy, 4),
            "timeline": timeline, "per_node": per_node}


# ---------------- 3. 仿真核心 ----------------
def simulate(dag, decision, nodes=NODES, p_tx=P_TX_END):
    """
    输入: dag + 卸载决策 {task_id: "end"/"edge"/"cloud"}
    输出: makespan(用户拿到全部结果的时刻) + 终端能耗(焦耳) + 时间线
    时序模型: start = max( 前驱完成+传输, 输入上传完成, 目标层空闲 )
    """
    for tid, loc in decision.items():
        assert loc in VALID_LOC, f"非法位置 {loc}"

    finish = {}            # task_id -> 完成时刻(含结果回传, 供依赖与 makespan 使用)
    node_free = {k: 0.0 for k in VALID_LOC}
    timeline = []
    end_exec = 0.0         # 终端执行时间合计
    end_tx = 0.0           # 终端参与收发的时间合计

    # 与 HEFT 用同一个 rank 优先级拓扑序, 保证决策构造与仿真回放的排队次序一致
    for tid in topo_order(dag, priority=rank_upward(dag)):
        loc = decision[tid]
        ready = 0.0
        # (a) 入口任务: 原始输入数据要从终端上传
        d_in = dag.get("input_data", {}).get(tid, 0.0)
        if d_in > 0 and loc != "end":
            tt = transfer_time("end", loc, d_in)
            end_tx += tt
            ready = max(ready, tt)
        # (b) 依赖约束: 前驱完成 + 中间数据传输
        for p in dag["pred"].get(tid, []):
            tt = transfer_time(decision[p], loc, dag["data_size"][(p, tid)])
            if "end" in (decision[p], loc):
                end_tx += tt
            ready = max(ready, finish[p] + tt)
        # (c) 资源约束: 同层任务排队串行
        exec_t = dag["workload"][tid] / nodes[loc]["f"]
        start = max(ready, node_free[loc])
        done = start + exec_t
        node_free[loc] = done
        if loc == "end":
            end_exec += exec_t
        # (d) 出口任务: 结果回传终端, 用户收到才算结束
        d_out = dag.get("output_data", {}).get(tid, 0.0)
        if d_out > 0 and loc != "end":
            tt = transfer_time(loc, "end", d_out)
            end_tx += tt
            done += tt
        finish[tid] = done
        timeline.append({"task": tid, "loc": loc, "start": round(start, 4),
                         "finish": round(start + exec_t, 4)})

    makespan = max(finish.values())
    energy = (end_exec * nodes["end"]["p_exec"]
              + (makespan - end_exec) * nodes["end"]["p_idle"]
              + end_tx * p_tx)
    per_node = {k: sum(1 for t in timeline if t["loc"] == k) for k in VALID_LOC}
    return {"makespan": round(makespan, 4), "energy": round(energy, 4),
            "timeline": timeline, "per_node": per_node}


# ---------------- 4. 演示 ----------------
if __name__ == "__main__":
    dag = gen_dag(n=10, p_edge=0.3, seed=42)
    n = dag["n"]
    plans = {
        "全在端执行": {i: "end" for i in range(n)},
        "全在边执行": {i: "edge" for i in range(n)},
        "全在云执行": {i: "cloud" for i in range(n)},
        "贪心(大任务→云)": {i: ("cloud" if dag["workload"][i] > 200 else "end")
                            for i in range(n)},
    }
    print(f"DAG 规模: {n} 任务, 入口输入数据: {dag['input_data']}")
    print(f"{'方案':<16}{'总时延(s)':>12}{'终端能耗(J)':>14}{'端/边/云':>12}")
    for name, dec in plans.items():
        r = simulate(dag, dec)
        dist = f"{r['per_node']['end']}/{r['per_node']['edge']}/{r['per_node']['cloud']}"
        print(f"{name:<16}{r['makespan']:>12.3f}{r['energy']:>14.3f}{dist:>12}")
