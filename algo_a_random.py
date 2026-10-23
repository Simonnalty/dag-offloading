# -*- coding: utf-8 -*-
"""
算法A: 随机卸载 (Random Offloading) —— baseline 下限

设计思想: 每个子任务在 {端, 边, 云} 中等概率随机选择, 不利用任何任务/环境信息。
适用条件: 无。它只作性能下限参照——任何有信息量的算法都应显著优于它。
实现流程: 遍历任务, 各随机选一个位置。
复杂度: 时间 O(V), 空间 O(V)。
负责人: (填组员名字)
"""

import random
from dag_simulator import VALID_LOC


def random_offload(dag, seed=0):
    """输入 dag, 返回卸载决策 {task_id: "end"|"edge"|"cloud"}"""
    rng = random.Random(seed)
    return {i: rng.choice(VALID_LOC) for i in range(dag["n"])}


if __name__ == "__main__":
    from dag_simulator import gen_dag, simulate
    dag = gen_dag(n=30, seed=42)
    r = simulate(dag, random_offload(dag, seed=42))
    print(f"[算法A 随机卸载] 30任务 seed=42: "
          f"makespan={r['makespan']}s, 终端能耗={r['energy']}J")
