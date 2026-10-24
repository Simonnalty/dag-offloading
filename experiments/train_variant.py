# -*- coding: utf-8 -*-
"""训练辅助脚本: 指定种子训练 DQN 并保存权重, 用于多种子选优"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from algo_d_dqn import train

seed = int(sys.argv[1])
net = train(episodes=3000, dag_n=12, seed=seed)
torch.save(net.state_dict(), f"experiments/dqn_seed{seed}.pth")
print(f"seed={seed} 完成")
