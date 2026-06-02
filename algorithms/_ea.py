import numpy as np
from typing import List
import logging
from ._base import BaseOptimizer
from ._utils import get_init_samples
import random
from ._ea_operator import swap_mutation, insert_mutation, reversal_mutation, shuffle_mutation, shift_mutation
from ._ea_operator import order_crossover, pmx_crossover, cycle_crossover
import torch
from ._dqn_utils import Net, MLP, UniNet, UniNet_metadata

log = logging.getLogger(__name__)

class EA(BaseOptimizer):
    def __init__(
        self, dims, lb, ub, pop_size=20, init_sampler_type='permutation', selection_type='elite',
        mutation_type='swap', crossover_type='order', parent_selection_type='random',
        tournament_size=3, policy_path=None, num_segments=None, device='cpu'
    ):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.offspring_size = self.pop_size
        self.init_sampler_type = init_sampler_type
        self.selection_type = selection_type
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.parent_selection_type = parent_selection_type
        self.tournament_size = tournament_size

        self.population = []
        self.fitness = []

        self.device = device

        self.policy_type = None
        self.window_size = None
        if policy_path:
            print(f"Loading pre-trained policy from {policy_path}...")
            checkpoint = torch.load(policy_path, map_location=self.device)
            config = checkpoint['model_config']
            # 这个是之前的做法，没有统一所有 Benchmark 时使用的网络 <-  Edited by gzx 251108
            import os
            dirname = os.path.dirname(policy_path)
            self.policy_type = dirname
            if dirname == 'multi_envs_models':
                self.q_net = Net(
                    dims=checkpoint["model_config"]["dims"],
                    action_dim=checkpoint["model_config"]["action_dim"],
                    d_model=checkpoint["model_config"]["d_model"],
                    nhead=checkpoint["model_config"]["nhead"],
                    num_layers=checkpoint["model_config"]["num_layers"]
                )
            elif dirname == 'unified_envs_models':    
                # 统一后使用的网络 <- Edited by gzx 251108
                self.q_net = MLP(
                    obs_dim=config['obs_dim'],
                    n_actions=config['action_dim'],
                    hidden_sizes=config['hidden_sizes']
                )
            elif dirname == 'unified_envs_models_large':    
                # state 添加了向量化的相似性特征 <- Edited by gzx 251202
                self.num_segments = num_segments
                self.q_net = UniNet(
                    obs_dim=config['obs_dim'],
                    n_actions=config['action_dim'],
                    num_segments=num_segments,
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers']
                )
            elif dirname == 'unified_envs_models_large_with_metadata':
                # state 添加了向量化的相似性特征 + 元特征 <- Edited by gzx 251203
                self.num_segments = config['num_segments']
                self.q_net = UniNet_metadata(
                    obs_dim=config['obs_dim'],
                    n_actions=config['action_dim'],
                    num_segments=config['num_segments'],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers'],
                    hidden_sizes=[128, 64]
                )
            elif 'unified_envs_models_large_with_metadata_ngs' in dirname and 'window' not in dirname:
                # state 添加了向量化的相似性特征 + 元特征 + 允许所有合法动作 （带 action_mask）
                self.num_segments = config['num_segments']
                self.q_net = UniNet_metadata(
                    obs_dim=config['obs_dim'],
                    n_actions=config['action_dim'],
                    num_segments=config['num_segments'],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers'],
                    hidden_sizes=config['hidden_sizes']
                )
            elif 'unified_envs_models_large_with_metadata_ngs_withfullmask_window' in dirname:
                # state 添加了向量化的相似性特征 + 元特征 + 动作窗口
                self.num_segments = config['num_segments']
                self.window_size = config['window_size']
                self.q_net = UniNet_metadata(
                    obs_dim=config['obs_dim'],
                    n_actions=config['action_dim'],
                    num_segments=config['num_segments'],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers'],
                    hidden_sizes=config['hidden_sizes']
                )
                
            self.q_net.load_state_dict(checkpoint['model_state_dict'])

            self.q_net.to(self.device)  # 确保模型在正确的设备上
            self.q_net.eval()
            print(f"Q-net loaded and set to eval mode.")
        else:
            self.q_net = None

        self.constraints_dict = None

    def get_ckpt_dict(self):
        return {
            'population': self.population,
            'fitness': self.fitness,
        }

    def load_ckpt_dict(self, ckpt_dict):
        self.population = ckpt_dict['population']
        self.fitness = ckpt_dict['fitness']
        # fitness 越大越好
        best_idx = np.argmax(self.fitness)
        best_x = self.population[best_idx]
        best_y = self.fitness[best_idx]
        return best_x, best_y

    def _init_samples(self, init_sampler_type, n) -> List[np.ndarray]:
        points = get_init_samples(init_sampler_type, n, self.dims, self.lb, self.ub)
        points = [points[i] for i in range(len(points))]
        return points

    def _mutation(self, x, n_repeats=1):
        if self.mutation_type == 'mix':
            # 定义每个mutation类型及其对应的概率
            # mutation_options = [
            #     ('swap', 0.1),
            #     ('insert', 0.8),
            #     ('reversal', 0.04),
            #     ('shuffle', 0.04),
            #     ('shift', 0.02),
            # ]
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

        elif self.mutation_type == 'swap':
            next_x = swap_mutation(x, repeats=n_repeats)
        elif self.mutation_type == 'insert':
            next_x = insert_mutation(x, repeats=n_repeats)
        elif self.mutation_type == 'reversal':
            next_x = reversal_mutation(x, repeats=n_repeats)
        elif self.mutation_type == 'shuffle':
            next_x = shuffle_mutation(x, repeats=n_repeats)
        elif self.mutation_type == 'shift':
            next_x = shift_mutation(x)
        else:
            raise NotImplementedError
        return next_x

    def _crossover(self, x1, x2):
        if self.crossover_type == 'order':
            next_x = order_crossover(x1, x2)
        elif self.crossover_type == 'pmx':
            next_x = pmx_crossover(x1, x2)
        elif self.crossover_type == 'cycle':
            next_x = cycle_crossover(x1, x2)
        else:
            raise NotImplementedError
        return next_x

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

    def _tournament_select_parent(self, exclude_idx=None):
        candidate_indices = np.arange(len(self.population))
        if exclude_idx is not None and len(candidate_indices) > 1:
            candidate_indices = candidate_indices[candidate_indices != exclude_idx]

        k = min(self.tournament_size, len(candidate_indices))
        sampled_indices = np.random.choice(candidate_indices, k, replace=False)
        sampled_fitnesses = [self.fitness[idx] for idx in sampled_indices]
        return sampled_indices[int(np.argmax(sampled_fitnesses))]

    def _select_parent_indices(self):
        if self.parent_selection_type == 'random':
            return np.random.choice(range(len(self.population)), 2, replace=False)
        elif self.parent_selection_type == 'tournament':
            idx1 = self._tournament_select_parent()
            idx2 = self._tournament_select_parent(exclude_idx=idx1)
            return idx1, idx2
        else:
            raise NotImplementedError

    def to_tensor(self, x, device):
        # 确保输入数据转换为张量并移动到正确的设备
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=torch.float32, device=device)
        elif isinstance(x, list):
            return torch.as_tensor(np.array(x), dtype=torch.float32, device=device)
        else:
            return x.to(device) if hasattr(x, 'to') else x


    @torch.no_grad()
    def _neural_crossover_and_mutation(self, x1, x2, f1, f2, cur_x_best=None):
        from Environments import EarthBenchEnv
        from Environments import EarthBenchEnvUnified
        from Environments import EarthBenchEnvUnifiedNGS
        from Environments import EarthBenchEnvUnifiedNGSWindow

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
            if cur_x_best is not None:
                if 'unified_envs_models_large_with_metadata_ngs' in self.policy_type:
                    if 'withfullmask' in self.policy_type:
                        if self.window_size is not None:
                            env = EarthBenchEnvUnifiedNGSWindow(x1_int, x2_int, cur_x_best, self.dims, f1, f2, num_segments=self.num_segments, use_full_mask=True, window_size=self.window_size)
                        else:
                            env = EarthBenchEnvUnifiedNGS(x1_int, x2_int, cur_x_best, self.dims, f1, f2, num_segments=self.num_segments, use_full_mask=True)
                    else:
                        env = EarthBenchEnvUnifiedNGS(x1_int, x2_int, cur_x_best, self.dims, f1, f2, num_segments=self.num_segments, use_full_mask=False)
                else:
                    env = EarthBenchEnvUnified(x1_int, x2_int, cur_x_best, self.dims, f1, f2, num_segments=self.num_segments)
            else:
                env = EarthBenchEnv(x1_int, x2_int, self.dims, f1, f2)
            self.constraints_dict = env._get_constraints_dict()
        else:
            if cur_x_best is not None:
                if 'unified_envs_models_large_with_metadata_ngs' in self.policy_type:
                    if 'withfullmask' in self.policy_type:
                        if self.window_size is not None:
                            env = EarthBenchEnvUnifiedNGSWindow(x1_int, x2_int, cur_x_best, self.dims, f1, f2, num_segments=self.num_segments, constraints_dict=self.constraints_dict, use_full_mask=True, window_size=self.window_size)
                        else:
                            env = EarthBenchEnvUnifiedNGS(x1_int, x2_int, cur_x_best, self.dims, f1, f2, num_segments=self.num_segments, constraints_dict=self.constraints_dict, use_full_mask=True)
                    else:
                        env = EarthBenchEnvUnifiedNGS(x1_int, x2_int, cur_x_best, self.dims, f1, f2, num_segments=self.num_segments, constraints_dict=self.constraints_dict, use_full_mask=False)
                else:
                    env = EarthBenchEnvUnified(x1_int, x2_int, cur_x_best, self.dims, f1, f2, num_segments=self.num_segments, constraints_dict=self.constraints_dict)
            else:
                env = EarthBenchEnv(x1_int, x2_int, self.dims, f1, f2, self.constraints_dict)


        # out = env.reset()

        # s = out[0] if isinstance(out, tuple) else out
        # s = s.reshape(1, -1)

        # while True:
        #     logits, _ = self.q_net(s)
        #     action = logits.argmax(dim=1).item()
        #     s, reward, terminated, truncated, info = env.step(action)
        #     s = s.reshape(1, -1)
        #     if terminated or truncated:
        #         break

        out = env.reset()

        s = out[0] if isinstance(out, tuple) else out
        if isinstance(s, dict):
            obs = s.get('obs', s)
            action_mask = s.get('mask', None)
        else:
            obs = s
            action_mask = None
        obs = obs.reshape(1, -1)

        while True:
            logits, _ = self.q_net(obs)
            if action_mask is not None:
                # action_mask 中 False 的位置的 logit 设置为负无穷大
                logits[:, ~action_mask] = -float('inf')
            action = logits.argmax(dim=1).item()
            s, reward, terminated, truncated, info = env.step(action)
            if isinstance(s, dict):
                obs = s.get('obs', s)
                action_mask = s.get('mask', None)
            else:
                obs = s
                action_mask = None
            obs = obs.reshape(1, -1)
            if terminated or truncated:
                break
    
        # 确保输出在CPU上，以便与numpy数组兼容
        result = env.o_partial
        if torch.is_tensor(result):
            result = result.cpu().numpy()
        
        return np.array(result)

    def ask(self, n_repeats=1, use_rl=False, cur_x_best=None):
        offspring = []
        if len(self.population) == 0:
            offspring.extend(self._init_samples(self.init_sampler_type, self.pop_size))
        else:
            for _ in range(self.offspring_size):
                i, j = self._select_parent_indices()
                x1, x2, f1, f2 = self.population[i], self.population[j], self.fitness[i], self.fitness[j]
                if use_rl:
                    next_x = self._neural_crossover_and_mutation(x1, x2, f1, f2, cur_x_best)
                else:
                    next_x = self._crossover(x1, x2)
                    next_x = self._mutation(next_x, n_repeats)
                offspring.append(next_x)
                
        return offspring

    
    def tell(self, X: List[np.ndarray], Y: List):
        if len(self.population) == 0:
            self.population = X
            self.fitness = Y
        else:
            # print(len(self.population), len(X))
            self.population, self.fitness = self._selection(self.population, self.fitness, X, Y)
