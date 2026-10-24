# -*- coding: utf-8 -*-
"""
算法D: DQN 强化学习卸载（课题要求2 指定必须包含 1 种强化学习算法）

参考: Mnih et al. "Human-level control through deep reinforcement learning."
      Nature, 2015. (实现细节参考其前身 arXiv:1312.5602)

马尔可夫决策过程 (MDP) 设计:
  状态 (13 维, 全部归一化):
    当前任务: 计算量/w_max、是否入口任务、是否出口任务、调度进度
    每层 k∈{端,边,云}: 就绪时刻/t_ref、层空闲时刻/t_ref、前驱任务在该层的比例
  动作 (3 维): 把当前就绪任务卸载到 端/边/云
  奖励: 过程奖励 0; 终局奖励 -(makespan/t_ref + λ·energy/e_ref)
        (t_ref, e_ref 为"全在端执行"的参考量纲, 使时延与能耗同尺度可加)
  一个 episode: 按 rank 优先级拓扑序把一个 DAG 的所有子任务决策完。
  环境时序模型与 dag_simulator.simulate() 逐行一致(episode 末可断言相等)。

训练: 经验回放 + 目标网络 + ε-greedy, PyTorch(CPU)。
用法:
  python algo_d_dqn.py        # 训练小图(约3分钟) → 保存权重 → 与基线对比
负责人: (填组员名字)
"""

import random
from pathlib import Path

import torch
import torch.nn as nn

from dag_simulator import (NODES, topo_order, rank_upward,
                           transfer_time, transfer_duration,
                           VALID_LOC as LOCS)

P_TX = 1.2
INPUT_DIM = 13          # 静态环境状态维度
DYN_INPUT_DIM = 14      # 动态环境: 额外一维"当前带宽缩放"(归一化)
N_ACTIONS = 3
WEIGHTS_PATH = Path(__file__).resolve().parent / "algo_d_dqn_weights.pth"
DYN_WEIGHTS_PATH = Path(__file__).resolve().parent / "algo_d_dqn_dynamic.pth"


def scale_at(phases, t):
    """查询时刻 t 的带宽缩放(取 t_start <= t 的最后一段)"""
    cur = phases[0][1]
    for t_start, s in phases:
        if t_start <= t:
            cur = s
        else:
            break
    return cur


# ================= 1. 环境: 逐步复刻 simulate 的时序模型 =================
class DAGOffloadEnv:
    """
    把一个 DAG 的卸载决策过程封装成 MDP。
    phases=None 时为静态环境(与 simulate 一致, 13 维状态);
    提供 phases 时为时变带宽环境(时序与 simulate_dynamic 一致, 14 维状态,
    额外特征为决策时刻的带宽缩放), 策略可在线感知波动并自适应。
    """

    def __init__(self, dag, lam=0.5, phases=None):
        self.dag = dag
        self.lam = lam
        self.phases = phases
        self.order = topo_order(dag, priority=rank_upward(dag))
        self.t_ref = sum(dag["workload"].values()) / NODES["end"]["f"]
        self.e_ref = NODES["end"]["p_exec"] * self.t_ref
        self.w_max = max(dag["workload"].values())
        self.n = dag["n"]
        self.state_dim = INPUT_DIM + (1 if phases else 0)

    def _tt(self, t0, size, src, dst):
        """传输时长: 静态用 transfer_time, 动态按带宽调度逐段积分"""
        if self.phases is None:
            return transfer_time(src, dst, size)
        return transfer_duration(t0, size, src, dst, self.phases)

    def reset(self):
        self.finish = {}
        self.node_free = {k: 0.0 for k in LOCS}
        self.end_exec = 0.0
        self.end_tx = 0.0
        self.idx = 0
        self.decision = {}
        return self._state()

    def _state(self):
        """当前任务的 13 维状态向量(已归一化)"""
        i = self.order[self.idx]
        dag = self.dag
        preds = dag["pred"].get(i, [])
        f = [
            dag["workload"][i] / self.w_max,
            1.0 if not preds else 0.0,               # 入口任务
            1.0 if not dag["succ"].get(i) else 0.0,  # 出口任务
            self.idx / self.n,                        # 调度进度
        ]
        d_in = dag.get("input_data", {}).get(i, 0.0)
        layer_feats = []
        for k in LOCS:
            ready = 0.0
            if d_in > 0 and k != "end":
                ready = transfer_time("end", k, d_in) if self.phases is None \
                    else self._tt(0.0, d_in, "end", k)
            for p in preds:
                tt = self._tt(self.finish[p], dag["data_size"][(p, i)],
                              self.decision.get(p), k) if self.phases is not None \
                    else transfer_time(self.decision.get(p), k, dag["data_size"][(p, i)])
                ready = max(ready, self.finish[p] + tt)
            frac = (sum(1 for p in preds if self.decision.get(p) == k) / len(preds)
                    if preds else 0.0)
            layer_feats += [ready / self.t_ref,
                            self.node_free[k] / self.t_ref,
                            frac]
        base = f + layer_feats
        if self.phases is not None:
            t_now = max(self.node_free.values())
            base.append(scale_at(self.phases, t_now) / 1.5)   # 归一化到 ~[0.33,1]
        return base

    def step(self, action):
        """action∈{0,1,2} → 端/边/云。返回 (下一状态, 奖励, 是否结束, 信息)"""
        i = self.order[self.idx]
        k = LOCS[action]
        dag = self.dag

        # --- 时序推演: 与 simulate/simulate_dynamic 逐行一致 ---
        ready = 0.0
        d_in = dag.get("input_data", {}).get(i, 0.0)
        if d_in > 0 and k != "end":
            tt = self._tt(0.0, d_in, "end", k)
            self.end_tx += tt
            ready = max(ready, tt)
        for p in dag["pred"].get(i, []):
            tt = self._tt(self.finish[p], dag["data_size"][(p, i)], self.decision[p], k)
            if "end" in (self.decision[p], k):
                self.end_tx += tt
            ready = max(ready, self.finish[p] + tt)
        t_exec = dag["workload"][i] / NODES[k]["f"]
        start = max(ready, self.node_free[k])
        exec_fin = start + t_exec
        self.node_free[k] = exec_fin
        if k == "end":
            self.end_exec += t_exec
        done_time = exec_fin
        d_out = dag.get("output_data", {}).get(i, 0.0)
        if d_out > 0 and k != "end":
            tt = self._tt(exec_fin, d_out, k, "end")
            self.end_tx += tt
            done_time += tt
        self.finish[i] = done_time
        self.decision[i] = k

        self.idx += 1
        if self.idx < self.n:
            return self._state(), 0.0, False, {}
        # --- 终局: 与 simulate 相同的指标, 归一化后取负 ---
        makespan = max(self.finish.values())
        energy = (self.end_exec * NODES["end"]["p_exec"]
                  + (makespan - self.end_exec) * NODES["end"]["p_idle"]
                  + self.end_tx * P_TX)
        reward = -(makespan / self.t_ref + self.lam * energy / self.e_ref)
        return None, reward, True, {"makespan": makespan, "energy": energy}


# ================= 2. DQN 网络 / 经验回放 =================
class QNet(nn.Module):
    def __init__(self, hidden=64, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, cap=30000):
        self.buf, self.cap = [], cap

    def push(self, *transition):
        if len(self.buf) < self.cap:
            self.buf.append(transition)
        else:
            self.buf[random.randrange(self.cap)] = transition

    def sample(self, batch):
        return zip(*random.sample(self.buf, batch))

    def __len__(self):
        return len(self.buf)


# ================= 3. 训练 =================
def train(episodes=3000, lam=0.5, dag_n=12, seed=0,
          eps_decay_until=1800, target_sync=300, log_every=500,
          env_factory=None, input_dim=None, tag="DQN"):
    """
    在随机小 DAG 上训练 DQN。返回训练好的 Q 网络。
    env_factory: dag -> DAGOffloadEnv 的工厂; 传入动态环境工厂即训练动态策略。
    每 500 个 episode 在固定评估集上输出与 HEFT 的对比。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    dim = input_dim or INPUT_DIM
    mk_env = env_factory or (lambda d: DAGOffloadEnv(d, lam))
    policy_net, target_net = QNet(input_dim=dim), QNet(input_dim=dim)
    target_net.load_state_dict(policy_net.state_dict())
    opt = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    buf = ReplayBuffer()
    gamma, batch = 0.99, 128

    # 固定评估集
    eval_dags = [gen_dag_safe(dag_n, s) for s in range(8)]
    from algo_c_heft import heft

    def eval_net(net):
        objs = []
        with torch.no_grad():
            for dag in eval_dags:
                env = mk_env(dag)
                s = env.reset()
                done = False
                while not done:
                    q = net(torch.tensor(s, dtype=torch.float32))
                    _, r, done, _ = env.step(int(q.argmax()))
                    s = env._state() if not done else None
                objs.append(-r)   # r 为负的加权目标
        return sum(objs) / len(objs)

    heft_obj = None
    for ep in range(episodes):
        eps = max(0.05, 1.0 - 0.95 * ep / eps_decay_until)
        # 注: 曾试验混合规模(n 8~28)训练, 泛化反而显著变差(bootstrap 链过长、
        # 方差过大), 小图(n≈10-14)训练的策略向 n=30 泛化最好——实验记录见报告。
        dag = gen_dag_safe(dag_n + random.randint(-2, 2), random.randrange(10**6))
        env = mk_env(dag)
        s = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                a = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    a = int(policy_net(torch.tensor(s, dtype=torch.float32)).argmax())
            s2, r, done, _ = env.step(a)
            buf.push(s, a, r, s2, float(done))
            s = s2

        # 经验回放更新
        if len(buf) >= batch:
            states, actions, rewards, next_states, dones = buf.sample(batch)
            states = torch.tensor([x for x in states], dtype=torch.float32)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            non_final = torch.tensor([x is not None for x in next_states])
            next_states_nf = torch.tensor(
                [x for x in next_states if x is not None], dtype=torch.float32)

            q_sa = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = torch.zeros(batch)
                next_q[non_final] = target_net(next_states_nf).max(1).values
                target = rewards + gamma * (1 - dones) * next_q
            loss = nn.functional.smooth_l1_loss(q_sa, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (ep + 1) % target_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % log_every == 0 or ep == 0:
            if heft_obj is None:
                heft_obj = _heft_objective(eval_dags, lam, env_factory)
            obj = eval_net(policy_net)
            print(f"  [{tag}] episode {ep+1:>5}/{episodes}  ε={eps:.2f}  "
                  f"DQN加权目标={obj:.3f}  (HEFT={heft_obj:.3f}, 越小越好)", flush=True)

    return policy_net


def gen_dag_safe(n, seed):
    from dag_simulator import gen_dag
    n = max(6, min(n, 30))
    return gen_dag(n=n, p_edge=0.25, seed=seed)


def _heft_objective(eval_dags, lam, env_factory=None):
    from algo_c_heft import heft
    mk_env = env_factory or (lambda d: DAGOffloadEnv(d, lam))
    objs = []
    for dag in eval_dags:
        env = mk_env(dag)
        env.reset()
        done = False
        while not done:
            a = LOCS.index(heft_action(env))
            _, r, done, _ = env.step(a)
        objs.append(-r)
    return sum(objs) / len(objs)


def make_random_schedule(rng, period_choices=(2.0, 4.0, 6.0),
                         scale_choices=(0.5, 1.0, 1.5), horizon=120.0):
    """为动态环境训练/评估生成随机带宽调度(周期与各段缩放均随机)"""
    from dag_simulator import make_telegraph_phases
    period = rng.choice(period_choices)
    scales = [rng.choice(scale_choices) for _ in range(rng.randint(2, 4))]
    return make_telegraph_phases(period, scales, horizon)


def heft_action(env):
    """在环境当前状态下用 HEFT 的 EFT 准则选层(用于对比, 不训练)"""
    from algo_c_heft import _transfer, _ready_time
    i = env.order[env.idx]
    dag = env.dag
    best, best_eft = None, float("inf")
    for k in LOCS:
        start = max(_ready_time(dag, i, k, env.decision, env.finish),
                    env.node_free[k])
        done = start + dag["workload"][i] / NODES[k]["f"]
        d_out = dag.get("output_data", {}).get(i, 0.0)
        if d_out > 0 and k != "end":
            done += _transfer(k, "end", d_out)
        if done < best_eft:
            best, best_eft = k, done
    return best


# ================= 4. 推理: 用训练好的网络输出卸载决策 =================
def dqn_offload(dag, net=None, lam=0.5):
    """输入 dag, 返回 {task_id: 层}。net 为 None 时加载已保存的权重。"""
    if net is None:
        net = QNet()
        net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    net.eval()
    env = DAGOffloadEnv(dag, lam)
    s = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            a = int(net(torch.tensor(s, dtype=torch.float32)).argmax())
            s, _, done, _ = env.step(a)
    return dict(env.decision)


def dqn_dynamic_offload(dag, phases, net=None, lam=0.5):
    """
    动态环境推理: 策略在每一步观测当前带宽缩放(状态第 14 维), 在线自适应决策。
    net 为 None 时加载已保存的动态权重(algo_d_dqn_dynamic.pth)。
    """
    if net is None:
        net = QNet(input_dim=DYN_INPUT_DIM)
        net.load_state_dict(torch.load(DYN_WEIGHTS_PATH, map_location="cpu"))
    net.eval()
    env = DAGOffloadEnv(dag, lam, phases=phases)
    s = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            a = int(net(torch.tensor(s, dtype=torch.float32)).argmax())
            s, _, done, _ = env.step(a)
    return dict(env.decision)


# ================= 5. 演示: 训练 + 与基线对比 =================
if __name__ == "__main__":
    from dag_simulator import simulate
    from algo_a_random import random_offload
    from algo_b_greedy import greedy_offload
    from algo_c_heft import heft
    from algo_e_eheft import eheft

    LAM = 0.5
    print(f"训练 DQN (小图 n≈10-14, λ={LAM}, 约4分钟)...")
    net = train(episodes=3000, lam=LAM, dag_n=12)
    torch.save(net.state_dict(), WEIGHTS_PATH)
    print(f"权重已保存: {WEIGHTS_PATH.name}")

    # 在更大的、训练时没见过的 DAG 上评估泛化
    print(f"\n泛化评估 (n=30, 10个未见过的seed, 加权目标 = makespan + {LAM}·能耗):")
    print(f"{'算法':<12}{'时延均值':>10}{'能耗均值':>10}{'加权目标':>10}")
    test_dags = [gen_dag_safe(30, 1000 + s) for s in range(10)]
    plans = [("A-random", random_offload), ("B-greedy", greedy_offload),
             ("C-HEFT", heft), ("E-EHEFT", lambda d: eheft(d, alpha=0.5)),
             ("D-DQN", lambda d: dqn_offload(d, net, lam=LAM))]
    for name, fn in plans:
        ms, en, ob = [], [], []
        for dag in test_dags:
            r = simulate(dag, fn(dag))
            ms.append(r["makespan"]); en.append(r["energy"])
            ob.append(r["makespan"] + LAM * r["energy"])
        n = len(test_dags)
        print(f"{name:<12}{sum(ms)/n:>10.3f}{sum(en)/n:>10.3f}{sum(ob)/n:>10.3f}")
