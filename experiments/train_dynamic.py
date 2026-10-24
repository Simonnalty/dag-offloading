# -*- coding: utf-8 -*-
"""训练动态 DQN 策略: 指定种子, 保存到 experiments/dqn_dyn_seed{seed}.pth"""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from algo_d_dqn import train, DAGOffloadEnv, make_random_schedule

seed = int(sys.argv[1])
lam = 0.5
rng = random.Random(seed * 7919)

def factory(dag):
    """每个训练 episode 随机生成带宽调度(周期 2-6s, 缩放 0.5/1.0/1.5 随机组合)"""
    return DAGOffloadEnv(dag, lam, phases=make_random_schedule(rng))

net = train(episodes=3000, lam=lam, dag_n=12, seed=seed,
            env_factory=factory, input_dim=14, tag=f"DYN-s{seed}")
torch.save(net.state_dict(), f"experiments/dqn_dyn_seed{seed}.pth")
print(f"seed={seed} 动态训练完成")
