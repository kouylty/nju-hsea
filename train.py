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
# from tianshou.utils.net.common import Net

import torch.nn.functional as F
from algorithms._dqn_utils import Net

from tianshou.utils.space_info import SpaceInfo
from torch.utils.tensorboard import SummaryWriter

from utils import _get_env, _get_envs

from algorithms._utils import init_v3_sampler
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_dims = 2582
    default_epochs = 2000
    parser.add_argument("--task_name", type=str, default=f"EarthBenchEnv_{default_dims}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=default_epochs)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_envs", type=int, default=100)
    parser.add_argument("--test_envs", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--target_freq", type=int, default=20)
    parser.add_argument("--buffer_size", type=int, default=15000)
    parser.add_argument("--eps_train", type=float, default=0.1)
    parser.add_argument("--eps_test", type=float, default=0.05)
    parser.add_argument("--epoch_num_steps", type=int, default=default_dims * 10)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=default_dims * 2)
    # parser.add_argument("--hidden_sizes", type=int, nargs='*', default=[512, 256, 128])
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--save_path", type=str, default=f"multi_envs_models/dim{default_dims}_epoch{default_epochs}.pth")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return args

def _test(task_name, net, idx=None, p1=None, p2=None, use_local_search=False):
    from utils import _get_parents, local_search
    import itertools
    # from Environments import EarthBenchEnv_278 as EarthBenchEnv
    from Environments import EarthBenchEnv
    dims = int(task_name.split('_')[-1])
    if idx is not None:
        parents = _get_parents(dims)
        all_pairs = list(itertools.combinations(parents, 2))
        selected_pairs = [all_pairs[i] for i in idx]
        if use_local_search:
            p1, f1, p2, f2 = selected_pairs[0][0][0], selected_pairs[0][0][1], selected_pairs[0][1][0], selected_pairs[0][1][1]
            p1, f1 = local_search(p1, f1, dims=dims, epochs=200)
            p2, f2 = local_search(p2, f2, dims=dims, epochs=200)
            selected_pairs = [((p1, f1), (p2, f2))] * len(idx)

        # print(f"selected pairs fitness: {[ (p[0][1], p[1][1]) for p in selected_pairs] }")
        test_envs = [EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), dims, p[0][1], p[1][1], is_test=True, local_search=use_local_search) for p in selected_pairs]
    elif p1 is not None and p2 is not None:
        test_envs = [EarthBenchEnv(p1, p2, dims, is_test=True)]
    
    print(f"Start testing.")
    start_time = time.time()
    reward_list = []
    for i, env in enumerate(test_envs):
        f1, f2 = env.f1, env.f2
        obs, _ = env.reset()
        obs = obs.reshape(1, -1)
        while True:
            logits, _ = net(obs)
            action = logits.argmax(dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            obs = obs.reshape(1, -1)
            if terminated or truncated:
                break
        if use_local_search:
            offspring = obs.flatten()[-dims:]
            _, local_search_fitness = local_search(offspring, dims=dims, epochs=200)
            reward_list.append(local_search_fitness - max(f1, f2)) # 根据 RL 给出的解进行 local search，重新计算 reward
        else:
            reward_list.append(reward)
    print(f"Test ended! Avg reward: {np.mean(reward_list)} of {len(reward_list)} envs. Using time: {time.time() - start_time:.2f}s.")

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
    writer = SummaryWriter(f"log/dqn-earth_{curr_time}")
    writer.add_text("args", str(args))
    logger = ts.utils.TensorboardLogger(writer)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 构建向量环境
    different_envs = 10
    train_envs, idx = _get_envs(task_name, num_train_envs, different_envs=different_envs, env_type='Subproc', need_indices=True, use_local_search=False)
    # test_envs = _get_envs(task_name, num_test_envs, env_type='Dummy')
    test_envs = _get_envs(task_name, num_test_envs, different_envs=different_envs, env_type='Subproc', use_local_search=False)

    # 单个环境用于网络 shape 信息
    env = _get_env(task_name)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    # Q Net
    # hidden_sizes = args.hidden_sizes
    net = Net(dims=env.dims, action_dim=action_shape, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers).to(args.device)
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

    def test_fn(epoch: int, env_step: int | None) -> None:
        test_freq = 500
        if (epoch + 1) % test_freq == 0:
            print(f"Epoch #{epoch + 1} Testing.")
            _test(task_name, net, idx, use_local_search=True)
        
    def stop_fn(mean_rewards: float) -> bool:
        return False

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
            # "state_shape": state_shape,
            # "action_shape": action_shape,
            # "hidden_sizes": hidden_sizes,
            "dims": env.dims,
            "action_dim": action_shape,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
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
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

    # _test(task_name, net, p1=p1, p2=p2)
    # _test(task_name, net, idx=idx, use_local_search=True)

    # # 测试表现
    # collector = ts.data.Collector[CollectStats](algorithm, test_envs, exploration_noise=False)
    # collector.collect(n_episode=5, reset_before_collect=True)



if __name__ == "__main__":
    main()