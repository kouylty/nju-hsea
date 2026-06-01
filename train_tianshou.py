# dqn_custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import json
import time
import torch

import argparse
import tianshou as ts
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import CollectStats
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net

from tianshou.utils.space_info import SpaceInfo
from torch.utils.tensorboard import SummaryWriter

from utils import _get_env, _get_envs

from algorithms._utils import init_v3_sampler
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="EarthBenchEnv_278")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_envs", type=int, default=1)
    parser.add_argument("--test_envs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--target_freq", type=int, default=30)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--eps_train", type=float, default=0.1)
    parser.add_argument("--eps_test", type=float, default=0.05)
    parser.add_argument("--epoch_num_steps", type=int, default=300)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=10)
    parser.add_argument("--hidden_sizes", type=int, nargs='*', default=[512, 256, 128])
    parser.add_argument("--save_path", type=str, default="dqn_ckpt_v6_tmp.pth")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return args

def _test(task_name, net, idx=None, p1=None, p2=None):
    from utils import _get_parents
    import itertools
    if task_name == "EarthBenchEnv_278":
        from Environments import BaseEarthBenchEnv_278 as EarthBenchEnv
        if idx is not None:
            dims = 278
            parents = _get_parents(dims)
            all_pairs = list(itertools.combinations(parents, 2))
            selected_pairs = [all_pairs[i] for i in idx]
            print(f"selected pairs fitness: {[ (p[0][1], p[1][1]) for p in selected_pairs] }")
            test_envs = [EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), dims, p[0][1], p[1][1]) for p in selected_pairs]
        elif p1 is not None and p2 is not None:
            dims = 278
            test_envs = [EarthBenchEnv(p1, p2, dims)]
    else:
        raise NotImplementedError

    print(f"Start testing.")
    for i, env in enumerate(test_envs):
        obs, _ = env.reset()
        obs = obs.reshape(1, -1)
        while True:
            logits, _ = net(obs)
            action = logits.argmax(dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            obs = obs.reshape(1, -1)
            if terminated or truncated:
                print(f"Reward of {i}th env: {reward}")
                break
    print("Test ended!")

def main(args: argparse.Namespace = get_args()):
    task_name = args.task_name
    lr, epoch, batch_size = args.lr, args.epoch, args.batch_size
    num_train_envs, num_test_envs = args.train_envs, args.test_envs
    gamma, n_step, target_freq = args.gamma, args.n_step, args.target_freq
    buffer_size = args.buffer_size
    eps_train, eps_test = args.eps_train, args.eps_test
    epoch_num_steps, collection_step_num_env_steps = args.epoch_num_steps, args.collection_step_num_env_steps
    print(f"save_path: {args.save_path}, device: {args.device}")

    curr_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    writer = SummaryWriter(f"log_single/dqn-earth_{curr_time}")
    writer.add_text("args", str(args))
    logger = ts.utils.TensorboardLogger(writer)

    # 生成两个父代解，用于定义环境
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 构建向量环境
    train_envs, idx = _get_envs(task_name, num_train_envs, env_type='Dummy', need_indices=True)
    # test_envs = _get_envs(task_name, num_test_envs, env_type='Dummy')
    test_envs = train_envs
    # test_envs = train_envs

    # 单个环境用于网络 shape 信息
    env = _get_env(task_name)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    # Q 网络
    hidden_sizes = args.hidden_sizes
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=hidden_sizes).to(args.device)
    # print("Test before training:")
    # _test(task_name, net, idx)
    # print("Pre-Test ended, start training!\n")

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
    test_collector = ts.data.Collector[CollectStats](
        algorithm,
        test_envs,
        exploration_noise=True,
    )

    def stop_fn(mean_rewards: float) -> bool:
        return False

    result = algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collection_step_num_env_steps,
            test_step_num_episodes=num_test_envs,
            batch_size=batch_size,
            update_step_num_gradient_steps_per_sample=1 / collection_step_num_env_steps,
            stop_fn=stop_fn,
            logger=logger,
            test_in_train=True,
        )
    )
    print("Training finished:", result.best_reward)

    save_path = args.save_path
    checkpoint = {
        "model_state_dict": net.state_dict(),   # Q 网络参数
        "policy_state_dict": policy.state_dict(),  # policy 封装的参数 (可选)
        "model_config": {
            "state_shape": state_shape,
            "action_shape": action_shape,
            "hidden_sizes": hidden_sizes,
            "gamma": gamma,
            "n_step": n_step,
        },
        "train_config": {
            "epoch": epoch,
            "batch_size": batch_size,
            "lr": lr,
            "buffer_size": buffer_size,
        },
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

    # _test(task_name, net, p1=p1, p2=p2)
    _test(task_name, net, idx=idx)

    # # 测试表现
    # collector = ts.data.Collector[CollectStats](algorithm, test_envs, exploration_noise=False)
    # collector.collect(n_episode=5, reset_before_collect=True)



if __name__ == "__main__":
    main()