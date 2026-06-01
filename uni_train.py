# dqn_custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import json
import time
import torch
import torch.nn as nn

import argparse
import tianshou as ts
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import CollectStats
from tianshou.trainer import OffPolicyTrainerParams

import torch.nn.functional as F
from algorithms._dqn_utils import Net, MLP

from tianshou.utils.space_info import SpaceInfo
from torch.utils.tensorboard import SummaryWriter

from utils import _get_unified_env, _get_unified_envs

from algorithms._utils import init_v3_sampler
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_dims = 2538
    default_epochs = 2000
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=default_epochs)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_envs", type=int, default=100)
    parser.add_argument("--test_envs", type=int, default=100)
    # parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--target_freq", type=int, default=20)
    parser.add_argument("--buffer_size", type=int, default=15000)
    # parser.add_argument("--buffer_size", type=int, default=30000)
    parser.add_argument("--eps_train", type=float, default=0.1)
    parser.add_argument("--eps_test", type=float, default=0.05)
    parser.add_argument("--epoch_num_steps", type=int, default=default_dims * 10)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=default_dims * 2)
    parser.add_argument("--hidden_sizes", type=int, nargs='*', default=[128, 128, 64])
    parser.add_argument("--num_segments", type=int, default=None)
    parser.add_argument("--save_path", type=str, default=f"unified_envs_models_large/epoch{default_epochs}_10_partial_dims.pth")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace = get_args()):
    lr, epoch, batch_size = args.lr, args.epoch, args.batch_size
    num_train_envs, num_test_envs = args.train_envs, args.test_envs
    gamma, n_step, target_freq = args.gamma, args.n_step, args.target_freq
    buffer_size = args.buffer_size
    eps_train, eps_test = args.eps_train, args.eps_test
    epoch_num_steps, collection_step_num_env_steps = args.epoch_num_steps, args.collection_step_num_env_steps
    num_segments = args.num_segments
    print(f"save_path: {args.save_path}, device: {args.device}, num_segments: {num_segments}")

    curr_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    writer = SummaryWriter(f"log/dqn-earth_{curr_time}")
    writer.add_text("args", str(args))
    logger = ts.utils.TensorboardLogger(writer)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 构建向量环境
    different_envs_per_dim = 10
    print("Start creating train envs...")
    train_envs = _get_unified_envs(num_train_envs, different_envs_per_dim=different_envs_per_dim, num_segments=num_segments, env_type='Subproc')
    print("Train envs created! Start creating test envs...")
    # test_envs = _get_envs(task_name, num_test_envs, env_type='Dummy')
    # test_envs = _get_unified_envs(num_test_envs, different_envs_per_dim=different_envs_per_dim, env_type='Subproc')
    # print("Test envs created!")

    env = _get_unified_env(num_segments=num_segments)

    obs_dim = 9
    n_actions = 3
    hidden_sizes = args.hidden_sizes
    net = MLP(obs_dim, n_actions, hidden_sizes).to(args.device)

    optim = AdamOptimizerFactory(lr=lr)

    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        eps_training=eps_train,
        eps_inference=eps_test,
    )
    algorithm = ts.algorithm.DQN(
        policy=policy,
        optim=optim,
        gamma=gamma,
        n_step_return_horizon=n_step,
        target_update_freq=target_freq,
    ).to(args.device)

    train_collector = ts.data.Collector[CollectStats](
        algorithm,
        train_envs,
        ts.data.VectorReplayBuffer(buffer_size, num_train_envs),
        exploration_noise=True,
    )
    # test_collector = ts.data.Collector[CollectStats](
    #     algorithm,
    #     test_envs,
    #     exploration_noise=True,
    # )

    def stop_fn(mean_rewards: float) -> bool:
        return False

    print("Start training...")
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collection_step_num_env_steps,
            batch_size=batch_size,
            update_step_num_gradient_steps_per_sample=1 / collection_step_num_env_steps,
            logger=logger,
            # test_collector=test_collector,
            # test_step_num_episodes=num_test_envs,
            # test_fn=test_fn,
            # stop_fn=stop_fn,
            # test_in_train=True,
        )
    )
    print("Training finished:", result.best_reward)

    save_path = args.save_path
    checkpoint = {
        "model_state_dict": net.state_dict(),   # Q 网络参数
        "policy_state_dict": policy.state_dict(),  # policy 封装的参数 (可选)
        "model_config": {
            "obs_dim": obs_dim,
            "action_dim": n_actions,
            "hidden_sizes": hidden_sizes,
        },
        "train_config": {
            "epoch": epoch,
            "batch_size": batch_size,
            "lr": lr,
            "buffer_size": buffer_size,
        },
    }
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()