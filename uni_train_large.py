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
from algorithms._dqn_utils import Net, MLP, UniNet, UniNet_metadata

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
    default_epochs = 3000
    n_train_envs = 200
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=default_epochs)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_envs", type=int, default=n_train_envs)
    parser.add_argument("--test_envs", type=int, default=n_train_envs)
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
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_segments", type=int, default=32)
    parser.add_argument("--hidden_sizes", type=list, default=[512, 512])
    parser.add_argument("--save_path", type=str, default=f"unified_envs_models_large_with_metadata_ngs_withfullmask/epoch{default_epochs}_trainenvs{n_train_envs}_20_partial_dims135_numsegment32_dmodel64.pth")
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
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
    writer = SummaryWriter(f"log_uni_vec_ngs_withfullmask/train_envs{num_train_envs}_135_dmodel{args.d_model}_{curr_time}")
    writer.add_text("args", str(args))
    logger = ts.utils.TensorboardLogger(writer)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 构建向量环境
    different_envs_per_dim = 20
    print("Start creating train envs...")
    train_envs = _get_unified_envs(num_train_envs, different_envs_per_dim=different_envs_per_dim, num_segments=num_segments, env_type='Subproc')
    print("Train envs created! Start creating test envs...")
    # test_envs = _get_envs(task_name, num_test_envs, env_type='Dummy')
    # test_envs = _get_unified_envs(num_test_envs, different_envs_per_dim=different_envs_per_dim, env_type='Subproc')
    # print("Test envs created!")

    env = _get_unified_env(num_segments=num_segments)

    obs_dim = env.obs_dim
    n_actions = env.action_dim
    net = UniNet_metadata(obs_dim, n_actions, num_segments=num_segments, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers).to(args.device)

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

    def train_fn(epoch: int, env_step: int) -> None:                
        # 计算分母，避免除零错误
        # total_actions = sum(train_collector.env.get_env_attr('a1_counter')) + sum(train_collector.env.get_env_attr('a2_counter')) + sum(train_collector.env.get_env_attr('a3_counter'))
        total_actions = sum(train_collector.env.get_env_attr('counter'))
        
        logger.write(
            "train/env_step",  # step_type: 命名空间
            env_step,  # step: 当前步数
        {
            # # data: 包含所有要记录的标量的字典
            # "a1_ratio": sum(train_collector.env.get_env_attr('a1_counter')) / total_actions if total_actions > 0 else 0,
            # "a2_ratio": sum(train_collector.env.get_env_attr('a2_counter')) / total_actions if total_actions > 0 else 0,
            # "a3_ratio": sum(train_collector.env.get_env_attr('a3_counter')) / total_actions if total_actions > 0 else 0,
            # "a1_fail_ratio": sum(train_collector.env.get_env_attr('a1_fail_counter')) / sum(train_collector.env.get_env_attr('a1_counter')) 
            #                         if sum(train_collector.env.get_env_attr('a1_counter')) > 0 else 0,
            # "a2_fail_ratio": sum(train_collector.env.get_env_attr('a2_fail_counter')) / sum(train_collector.env.get_env_attr('a2_counter')) 
            #                         if sum(train_collector.env.get_env_attr('a2_counter')) > 0 else 0,
            "invalid_ratio": sum(train_collector.env.get_env_attr('invalid_counter')) / total_actions if total_actions > 0 else 0,
            "action_out_of_range_ratio": sum(train_collector.env.get_env_attr('action_out_of_range_counter')) / total_actions if total_actions > 0 else 0,
        }
        )

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
            train_fn=train_fn,
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
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "num_segments": num_segments,
            "hidden_sizes": args.hidden_sizes,
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