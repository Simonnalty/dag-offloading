# -*- coding: utf-8 -*-
"""
精确基准的自校验 (改了 dag_simulator.py 或 algo_f_exact.py 之后必跑)

校验内容:
  [1] 快速评分 evaluate() 与官方 simulate() 在随机实例/随机决策上逐点一致;
  [2] 枚举得到的 Pareto 前沿确实非支配, 且覆盖全部决策(没有漏掉更优解);
  [3] exact_opt() 与"直接用 simulate 暴力扫描"得到的最优值一致 (独立路径);
  [4] 前沿点的决策回放后能复现该点坐标;
  [5] 规模保护: n > 12 拒绝枚举。

用法: python verify_exact.py      (全部通过时退出码 0)
"""

import itertools
import random
import sys

from dag_simulator import gen_dag, simulate
from algo_f_exact import (LOCS, MAX_EXACT_N, _eval_combo, _playback_order,
                          evaluate, exact_front, exact_opt, is_dominated,
                          on_front, objective)

PASS = 0


def check(name, cond=True, detail=""):
    global PASS
    assert cond, f"[FAIL] {name} {detail}"
    PASS += 1
    print(f"[PASS] {name}")


def test_evaluate_matches_simulator():
    """[1] 快速评分必须与官方仿真器完全一致"""
    rng = random.Random(20261110)
    cases = 0
    for n in (5, 8, 13, 20, 30):
        for _ in range(12):
            dag = gen_dag(n=n, p_edge=0.2, seed=rng.randrange(10 ** 6))
            decision = {i: rng.choice(LOCS) for i in range(n)}
            ms_fast, en_fast = evaluate(dag, decision)
            r = simulate(dag, decision)
            assert abs(ms_fast - r["makespan"]) < 1e-9, (n, ms_fast, r["makespan"])
            assert abs(en_fast - r["energy"]) < 1e-9, (n, en_fast, r["energy"])
            cases += 1
    check(f"[1] evaluate() 与 simulate() 一致 ({cases} 个随机实例/决策)")


def test_front_is_nondominated_and_complete():
    """[2] 前沿非支配性 + 完整性(随机抽样点都应被前沿支配或落在前沿上)"""
    for n in (5, 7):
        for seed in range(3):
            dag = gen_dag(n=n, p_edge=0.2, seed=seed)
            front = exact_front(dag)
            assert front, "前沿为空"
            # 互不支配
            for a in range(len(front)):
                for b in range(a + 1, len(front)):
                    pa, pb = front[a], front[b]
                    assert not (pa["makespan"] <= pb["makespan"]
                                and pa["energy"] <= pb["energy"]), "前沿内存在支配关系"
            # 排序: makespan 升序, energy 严格降序
            for a in range(len(front) - 1):
                assert front[a]["makespan"] <= front[a + 1]["makespan"] + 1e-9
                assert front[a]["energy"] > front[a + 1]["energy"]
            # 随机 400 个决策: 要么落在前沿, 要么被前沿支配
            rng = random.Random(1000 + seed)
            for _ in range(400):
                combo = tuple(rng.randrange(3) for _ in range(n))
                ms, en = _eval_combo(dag, combo, _playback_order(dag))
                assert on_front(ms, en, front) or is_dominated(ms, en, front), \
                    f"漏掉更优解: n={n} seed={seed} ({ms},{en})"
    check("[2] 前沿非支配、排序正确、随机抽样点无遗漏 (n=5,7 各 3 seed)")


def test_opt_matches_bruteforce_simulate():
    """[3] exact_opt 与 simulate 暴力扫描的最优值一致(独立实现路径)"""
    for n in (4, 5, 6):
        for seed in range(4):
            dag = gen_dag(n=n, p_edge=0.3, seed=seed)
            best = exact_opt(dag, lam=0.5)
            brute = float("inf")
            for combo in itertools.product(range(3), repeat=n):
                decision = {i: LOCS[c] for i, c in enumerate(combo)}
                r = simulate(dag, decision)
                brute = min(brute, objective(r["makespan"], r["energy"], 0.5))
            assert abs(best["objective"] - round(brute, 4)) < 1e-6, \
                f"n={n} seed={seed}: {best['objective']} vs {brute}"
    check("[3] exact_opt() 与暴力扫描最优值一致 (n=4,5,6 × 4 seed)")


def test_front_decision_replays():
    """[4] 前沿点的决策回放后必须复现该点坐标"""
    dag = gen_dag(n=9, p_edge=0.15, seed=3)
    for p in exact_front(dag):
        r = simulate(dag, p["decision"])
        assert abs(r["makespan"] - p["makespan"]) < 1e-9
        assert abs(r["energy"] - p["energy"]) < 1e-9
        assert sum(p["per_node"].values()) == dag["n"]
    check("[4] 前沿点决策回放一致 (n=9 全部前沿点)")


def test_size_guard():
    """[5] 超规模时明确拒绝, 防止误用导致指数爆炸"""
    dag = gen_dag(n=MAX_EXACT_N + 1, p_edge=0.1, seed=0)
    try:
        exact_front(dag)
    except ValueError:
        check(f"[5] n={MAX_EXACT_N + 1} 正确拒绝枚举 (上限 {MAX_EXACT_N})")
    else:
        raise AssertionError("[FAIL] 超规模未拒绝")


def main():
    print("=" * 64)
    print("精确基准自校验: evaluate / 前沿 / 最优值 / 规模保护")
    print("=" * 64)
    test_evaluate_matches_simulator()
    test_front_is_nondominated_and_complete()
    test_opt_matches_bruteforce_simulate()
    test_front_decision_replays()
    test_size_guard()
    print("-" * 64)
    print(f"全部 {PASS} 项通过.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
