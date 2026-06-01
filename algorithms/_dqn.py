import numpy as np
from typing import List
import logging
from ._base import BaseOptimizer
from ._utils import get_init_samples
import random
import torch
import time
# from ._dqn_utils import MLP
from ._dqn_utils import Net
from ._ea_operator import swap_mutation, insert_mutation, reversal_mutation, shuffle_mutation, shift_mutation
from ._ea_operator import order_crossover, pmx_crossover, cycle_crossover

log = logging.getLogger(__name__)


class DQN(BaseOptimizer):
    def __init__(
        self, dims, lb=None, ub=None, pop_size=20, init_sampler_type='permutation', selection_type='elite',
        policy_path=None, device='cpu'
    ):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.offspring_size = self.pop_size
        self.init_sampler_type = init_sampler_type
        self.selection_type = selection_type
        self.device = device
        print(f"device: {device}")

        if policy_path:
            print(f"Loading pre-trained policy from {policy_path}...")
            checkpoint = torch.load(policy_path, map_location=self.device)

            # 这里假设已经网络已经提前训练好了，直接加载即可
            # 之后也可以实现不用训练，直接随机初始化的版本
            # 提取模型配置参数
            config = checkpoint['model_config']

            # # 用正确的参数创建模型
            # self.q_net = MLP(
            #     obs_dim=config['obs_dim'],
            #     n_actions=config['n_actions'],
            #     hidden=config['hidden']
            # )
            # Using the tianshou framework
            # self.q_net = Net(state_shape=config['state_shape'], action_shape=config['action_shape'], hidden_sizes=config["hidden_sizes"])
            self.q_net = Net(
                dims=checkpoint["model_config"]["dims"],
                action_dim=checkpoint["model_config"]["action_dim"],
                d_model=checkpoint["model_config"]["d_model"],
                nhead=checkpoint["model_config"]["nhead"],
                num_layers=checkpoint["model_config"]["num_layers"]
            )
            self.q_net.load_state_dict(checkpoint['model_state_dict'])

            self.q_net.to(self.device)  # 确保模型在正确的设备上
            self.q_net.eval()
            print(f"Q-net loaded and set to eval mode.")
        else:
            # self.q_net = MLP(obs_dim=self.dims*3, n_actions=3, hidden=256)
            # self.q_net = Net(state_shape=config['state_shape'], action_shape=config['action_shape'], hidden_sizes=config["hidden_sizes"])
            self.q_net = Net(
                dims=self.dims * 3 + 2,
                action_dim=3,
                d_model=64,
                nhead=4,
                num_layers=3
            ).to(self.device)
        
        self.constraints_dict = None

        self.population = []
        self.fitness = []
        self.epoch_cnt = 0

    def _init_samples(self, init_sampler_type, n) -> List[np.ndarray]:
        points = get_init_samples(init_sampler_type, n, self.dims, self.lb, self.ub)
        points = [points[i] for i in range(len(points))]
        return points
    
    def to_tensor(self, x, device):
        # 确保输入数据转换为张量并移动到正确的设备
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=torch.float32, device=device)
        elif isinstance(x, list):
            return torch.as_tensor(np.array(x), dtype=torch.float32, device=device)
        else:
            return x.to(device) if hasattr(x, 'to') else x

    @torch.no_grad()
    def _neural_crossover_and_mutation(self, x1, x2, f1, f2):
        from Environments import EarthBenchEnv

        # 确保输入数据在正确的设备上
        x1_tensor = self.to_tensor(x1, self.device)
        x2_tensor = self.to_tensor(x2, self.device)
        
        # 转换为numpy数组并确保为整数类型
        def convert_to_int_array(data):
            if torch.is_tensor(data):
                data = data.cpu().numpy()
            # 根据需求选择合适的转换方式
            return data.astype(np.int32)  # 或者 np.int64
        
        x1_int = convert_to_int_array(x1_tensor)
        x2_int = convert_to_int_array(x2_tensor)
        
        if self.constraints_dict == None:
            env = EarthBenchEnv(x1_int, x2_int, self.dims, f1, f2)
            self.constraints_dict = env._get_constraints_dict()
        else:
            env = EarthBenchEnv(x1_int, x2_int, self.dims, f1, f2, self.constraints_dict)

        out = env.reset()

        s = out[0] if isinstance(out, tuple) else out
        s = s.reshape(1, -1)

        while True:
            logits, _ = self.q_net(s)
            action = logits.argmax(dim=1).item()
            s, reward, terminated, truncated, info = env.step(action)
            s = s.reshape(1, -1)
            if terminated or truncated:
                break
        
        # 确保输出在CPU上，以便与numpy数组兼容
        result = env.o_partial
        if torch.is_tensor(result):
            result = result.cpu().numpy()
        
        return np.array(result)

    def _selection(self, parents, parents_fit, offspring, offspring_fit, kappa=0.001):
        if self.selection_type == 'elite':
            all_individual = parents + offspring
            all_fitness = parents_fit + offspring_fit
            indices = np.argsort(all_fitness)[-self.pop_size: ]

            next_generation = [all_individual[idx] for idx in indices]
            next_fitness = [all_fitness[idx] for idx in indices]
            return next_generation, next_fitness
        elif self.selection_type == 'rank_based_prioritized':
            # Reference: Neural Genetic Search in Discrete Spaces
            all_individual = parents + offspring
            all_fitness = parents_fit + offspring_fit
            pop_size = self.pop_size
            N = len(all_individual)

            sorted_indices = np.argsort(all_fitness)[::-1]
            ranks = np.empty(N, dtype=int)
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank
            
            probs = 1.0 / (kappa * N + ranks)
            probs /= probs.sum()

            indices = np.random.choice(N, size=pop_size, replace=False, p=probs)

            next_generation = [all_individual[i] for i in indices]
            next_fitness = [all_fitness[i] for i in indices]
            return next_generation, next_fitness
        else:
            raise NotImplementedError
    
    def ask(self, n_repeats=1, use_rl=True):
        offspring = []
        if len(self.population) == 0:
            offspring.extend(self._init_samples(self.init_sampler_type, self.pop_size))
        else:
            for _ in range(self.offspring_size):
                i, j = np.random.choice(range(self.pop_size), 2, replace=False)
                x1, x2, f1, f2 = self.population[i], self.population[j], self.fitness[i], self.fitness[j]
                if use_rl:
                    next_x = self._neural_crossover_and_mutation(x1, x2, f1, f2)
                else:
                    x = pmx_crossover(x1, x2)
                    mutation_options = [
                        ('swap', 0.1),
                        ('insert', 0.8),
                        ('reversal', 0.04),
                        ('shuffle', 0.04),
                        ('shift', 0.02),
                    ]
                    
                    mutation_types, probabilities = zip(*mutation_options)
                    selected_mutation = random.choices(mutation_types, probabilities)[0]

                    if selected_mutation == 'swap':
                        next_x = swap_mutation(x, repeats=n_repeats)
                    elif selected_mutation == 'insert':
                        next_x = insert_mutation(x, repeats=n_repeats)
                    elif selected_mutation == 'reversal':
                        next_x = reversal_mutation(x, repeats=n_repeats)
                    elif selected_mutation == 'shuffle':
                        next_x = shuffle_mutation(x, repeats=n_repeats)
                    elif selected_mutation == 'shift':
                        next_x = shift_mutation(x)
                    else:
                        raise NotImplementedError
                    # next_x = insert_mutation(next_x)
                offspring.append(next_x)
        self.epoch_cnt += 1
        return offspring

    
    def tell(self, X: List[np.ndarray], Y: List):
        if len(self.population) == 0:
            self.population = X
            self.fitness = Y
        else:
            self.population, self.fitness = self._selection(self.population, self.fitness, X, Y)