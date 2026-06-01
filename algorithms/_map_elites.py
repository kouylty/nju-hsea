import numpy as np
from typing import List, Optional
import logging 
from ._base import BaseOptimizer
from ._utils import get_init_samples, kendall_ranking_correlation
from ._qd_utils import Archive
import random
from ._dqn_utils import Net, MLP, UniNet, UniNet_metadata
from ._ea_operator import swap_mutation, insert_mutation, reversal_mutation, shuffle_mutation, shift_mutation
from ._ea_operator import order_crossover, pmx_crossover, cycle_crossover
import time
import torch
# import ray

log = logging.getLogger(__name__)

class MAPElites(BaseOptimizer):
    def __init__(
        self, dims, lb, ub, archive_size,
        correlation_threshold: float = 0.1,
        pop_size = 20,
        init_sampler_type = 'permutation',
        selection_type = 'random',
        tournament_size = 3,
        mutation_type = 'swap',
        adaptive_phase_epochs = 2000,
        adaptive_prior_weight = 0.7,
        dynamic_correlation_threshold = False,
        dynamic_threshold_start = None,
        dynamic_threshold_end = None,
        dynamic_threshold_epochs = 2000,
        dynamic_threshold_warmup_size = None,
        crossover_type = 'order',
        # archive_init_ratio: float = 0.8,
        archive: Optional[Archive] = None,
        policy_path=None,
        num_segments=None,
        device = 'cpu'
    ) -> None:
        self.dims = dims 
        self.lb = lb 
        self.ub = ub
        self.correlation_threshold = correlation_threshold
        self.archive_size = archive_size
        self.offspring_size = pop_size
        # TODO import this from shell, this should be pop_size.
        self.init_sampler_type = init_sampler_type
        self.selection_type = selection_type
        self.tournament_size = tournament_size
        self.mutation_type = mutation_type
        self.adaptive_phase_epochs = adaptive_phase_epochs
        self.adaptive_prior_weight = adaptive_prior_weight
        self.dynamic_correlation_threshold = dynamic_correlation_threshold
        self.dynamic_threshold_start = dynamic_threshold_start
        self.dynamic_threshold_end = dynamic_threshold_end
        self.dynamic_threshold_epochs = dynamic_threshold_epochs
        self.dynamic_threshold_warmup_size = dynamic_threshold_warmup_size
        self.crossover_type = crossover_type
        self.device = device
        print(f"Using device: {self.device}")
        self.mutation_ops = ['swap', 'insert', 'reversal', 'shuffle', 'shift']
        self.mutation_attempts = {op: 0 for op in self.mutation_ops}
        self.mutation_successes = {op: 0 for op in self.mutation_ops}
        self.pending_mutation_types = []
        self.tell_count = 0
        
        if archive is not None:
            self.archive = archive
        else:
            self.archive = Archive(
                fitnesses=[],
                descriptors=[],
                individuals=[],
                archive_size=archive_size
            )

        self.policy_type = None
        self.window_size = None
        if policy_path:
            print(f"Loading pre-trained policy from {policy_path}...")
            checkpoint = torch.load(policy_path, map_location=self.device)
            config = checkpoint['model_config']
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
            'archive': self.archive,
            'mutation_attempts': self.mutation_attempts,
            'mutation_successes': self.mutation_successes,
            'tell_count': self.tell_count,
        }

    def load_ckpt_dict(self, ckpt_dict):
        self.archive = ckpt_dict['archive']
        self.mutation_attempts = ckpt_dict.get('mutation_attempts', self.mutation_attempts)
        self.mutation_successes = ckpt_dict.get('mutation_successes', self.mutation_successes)
        self.tell_count = ckpt_dict.get('tell_count', self.tell_count)
        # fitness 越大越好
        best_idx = np.argmax(self.archive.fitnesses)
        best_x = self.archive.individuals[best_idx]
        best_y = self.archive.fitnesses[best_idx]
        return best_x, best_y

    def _init_samples(self, init_sampler_type, n) -> List[np.ndarray]:
        points = get_init_samples(init_sampler_type, 
                                  n,
                                  self.dims,
                                  self.lb,
                                  self.ub)
        points = [points[i] for i in range(len(points))]
        return points 
    
    def _apply_mutation(self, x, mutation_type, n_repeats):
        if mutation_type == 'swap':
            return swap_mutation(x, repeats=n_repeats)
        elif mutation_type == 'insert':
            return insert_mutation(x, repeats=n_repeats)
        elif mutation_type == 'reversal':
            return reversal_mutation(x, repeats=n_repeats)
        elif mutation_type == 'shuffle':
            return shuffle_mutation(x, repeats=n_repeats)
        elif mutation_type == 'shift':
            return shift_mutation(x)
        else:
            raise NotImplementedError

    def _adaptive_prior_probs(self):
        progress = min(1.0, self.tell_count / max(1, self.adaptive_phase_epochs))
        if progress < 0.3:
            probs = {'swap': 0.08, 'insert': 0.32, 'reversal': 0.25, 'shuffle': 0.30, 'shift': 0.05}
        elif progress < 0.7:
            probs = {'swap': 0.10, 'insert': 0.50, 'reversal': 0.20, 'shuffle': 0.15, 'shift': 0.05}
        else:
            probs = {'swap': 0.25, 'insert': 0.60, 'reversal': 0.05, 'shuffle': 0.05, 'shift': 0.05}
        return np.array([probs[op] for op in self.mutation_ops], dtype=float)

    def _adaptive_success_probs(self):
        scores = []
        for op in self.mutation_ops:
            scores.append((self.mutation_successes[op] + 1.0) / (self.mutation_attempts[op] + 2.0))
        scores = np.array(scores, dtype=float)
        return scores / scores.sum()

    def _select_adaptive_mutation(self):
        prior_probs = self._adaptive_prior_probs()
        success_probs = self._adaptive_success_probs()
        probs = self.adaptive_prior_weight * prior_probs + (1 - self.adaptive_prior_weight) * success_probs
        probs = probs / probs.sum()
        return random.choices(self.mutation_ops, weights=probs, k=1)[0]

    def _mutation(self, x, n_repeats):
        if self.mutation_type == 'mix':
            # 定义每个mutation类型及其对应的概率
            mutation_options = [
                ('swap', 0.1),
                ('insert', 0.7),
                ('reversal', 0.05),
                ('shuffle', 0.1),
                ('shift', 0.05),
            ]
            
            mutation_types, probabilities = zip(*mutation_options)
            selected_mutation = random.choices(mutation_types, probabilities)[0]
            next_x = self._apply_mutation(x, selected_mutation, n_repeats)
        elif self.mutation_type == 'adaptive_mix':
            selected_mutation = self._select_adaptive_mutation()
            next_x = self._apply_mutation(x, selected_mutation, n_repeats)
        else:
            selected_mutation = self.mutation_type
            next_x = self._apply_mutation(x, selected_mutation, n_repeats)
        return next_x, selected_mutation

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
    

    def _update_desc(self, offspring):
        if len(self.archive) == 0:
            self.archive.descriptors = []
            return
        
        try:
            archive_features = [individual for individual in self.archive]
            # archive_tensor = torch.tensor(archive_features, dtype=torch.float32, device="cuda")
            archive_features = np.array(archive_features)   # 一次性拼成 ndarray
            archive_tensor = torch.from_numpy(archive_features).to(dtype=torch.float32, device=self.device) # (offspring_size, dim)
        except AttributeError:
            raise RuntimeError("Archive individuals must have a .values attribute containing feature vectors")
        except Exception as e:
            raise RuntimeError(f"Failed to convert archive features to tensor: {e}")
                
        if not isinstance(offspring, torch.Tensor):
            offspring = torch.tensor(offspring, dtype=torch.float32, device=self.device)
        if offspring.dim() == 1:
            offspring = offspring.unsqueeze(0)  # (1, c)
        
        # time1 = time.time()
        scores = kendall_ranking_correlation(
            supports=offspring,  # (1, c)
            queries=archive_tensor,  # (n, c)
            device=self.device
        )  # output.shape (n, 1)
        # print(f"Kendall correlation time: {time.time() - time1:.2f}s")
        self.archive.descriptors = scores.squeeze(-1).cpu().tolist()

    def _if_add(self, fitness):
        l = len(self.archive)
        correlation_threshold = self._current_correlation_threshold()
        if l == 0:
            return True, -1 
        elif l == 1:
            if self.archive.descriptors[0] <= correlation_threshold:
                return True, -1
            elif fitness >= self.archive.fitnesses[0]:
                return True, 0
            else:
                return False, -1
        else:
            max_corr = np.max(self.archive.descriptors)
            max_index = np.argmax(self.archive.descriptors)
            if max_corr <= correlation_threshold:
                return True, -1 
            elif fitness >= self.archive.fitnesses[max_index]:
                return True, max_index
            else:
                return False, -1

    def _current_correlation_threshold(self):
        if not self.dynamic_correlation_threshold:
            return self.correlation_threshold
        warmup_size = self.dynamic_threshold_warmup_size
        if warmup_size is None:
            warmup_size = min(self.archive_size, max(3, self.offspring_size))
        if len(self.archive) < warmup_size:
            return self.correlation_threshold
        start = self.dynamic_threshold_start
        if start is None:
            start = self.correlation_threshold * 0.9
        end = self.dynamic_threshold_end
        if end is None:
            end = self.correlation_threshold
        progress = min(1.0, self.tell_count / max(1, self.dynamic_threshold_epochs))
        return start + (end - start) * progress

    def _survival(self, offsprings, offsprings_fit, mutation_types=None):
        if mutation_types is None:
            mutation_types = [None] * len(offsprings)
        for offspring, fitness, mutation_type in zip(offsprings, offsprings_fit, mutation_types):
            # start_time = time.time()
            # 更新 当前子代解 相对于 当前 archive 中的所有个体的 Kendall 系数
            self._update_desc(offspring)  # Time-consuming for high-dim solutions.
            # 根据 当前子代解的fitness 确定这个子代解是否需要加入 archive
            add, index = self._if_add(fitness)
            if mutation_type in self.mutation_attempts:
                self.mutation_attempts[mutation_type] += 1
                if add:
                    self.mutation_successes[mutation_type] += 1
            if add:
                if index < 0:
                    self.archive.add_to_archive(offspring, fitness)
                else:
                    self.archive.update(offspring, fitness, index)
            # end_time = time.time()
            # log.info(f"One offspring, time: {end_time - start_time:.2f}s.")           

    def _selection(self, selection_type='random'):
        l = len(self.archive)
        if selection_type == 'random':
            idx1, idx2 = np.random.choice(l, 2, replace=False)
        elif selection_type == 'tournament':
            idx1 = self._tournament_select()
            idx2 = self._tournament_select(exclude_idx=idx1)
        else:
            raise NotImplementedError

        return self.archive.individuals[idx1], self.archive.individuals[idx2], idx1, idx2

    def _tournament_select(self, exclude_idx=None):
        candidate_indices = np.arange(len(self.archive))
        if exclude_idx is not None and len(candidate_indices) > 1:
            candidate_indices = candidate_indices[candidate_indices != exclude_idx]

        k = min(self.tournament_size, len(candidate_indices))
        sampled_indices = np.random.choice(candidate_indices, k, replace=False)
        sampled_fitnesses = [self.archive.fitnesses[idx] for idx in sampled_indices]
        return sampled_indices[int(np.argmax(sampled_fitnesses))]

    def _update_archive(self):
        if len(self.archive) > self.archive_size:
            self.archive.individuals = self.archive.individuals[-self.archive_size:]
            self.archive.fitnesses = self.archive.fitnesses[-self.archive_size:]
            self.archive.descriptors = self.archive.descriptors[-self.archive_size:]

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
                env = EarthBenchEnv(x1_int, x2_int, self.dims, f1, f2) # 这里时间开销比较大，每次需要0.1-0.2s
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
        offsprings = []
        self.pending_mutation_types = []
        if len(self.archive) <= 2:
            offsprings.extend(
                self._init_samples(self.init_sampler_type,
                                   2))
            self.pending_mutation_types.extend([None] * len(offsprings))
        else:
            for _ in range(self.offspring_size):
                x1, x2, idx1, idx2 = self._selection(self.selection_type)
                if use_rl:
                    x_nxt = self._neural_crossover_and_mutation(x1, x2,
                                                               self.archive.fitnesses[idx1],
                                                               self.archive.fitnesses[idx2], cur_x_best)
                    self.pending_mutation_types.append(None)
                else:
                    x_nxt = self._crossover(x1, x2)
                    x_nxt, mutation_type = self._mutation(x_nxt, n_repeats)
                    self.pending_mutation_types.append(mutation_type)
                offsprings.append(x_nxt)

        return offsprings, self.archive

    def tell(self, X: List[np.ndarray], Y: List):
        # if len(self.archive) <= self.archive_init_size:
        #     if len(X) + len(self.archive) <= self.archive_init_size:
        #         self.archive.add_to_archive(X, Y)
        #     else:
        #         process_mask = np.ones(len(X))
        #         idx = np.random.choice(range(len(X)),
        #                                self.archive_init_size - len(self.archive),
        #                                replace=False)
        #         process_mask[idx] = 0
                
        #         self.archive.add_to_archive(
        #             np.array(X)[idx], 
        #             np.array(Y)[idx])
                
        #         X_to_process = np.array(X)[np.where(process_mask == 0)[0]]
        #         Y_to_process = np.array(Y)[np.where(process_mask == 0)[0]]

        # else:
        #     X_to_process = np.array(X)
        #     Y_to_process = np.array(Y)
        
        # if X_to_process is not None:
        #     self._survival(X_to_process, Y_to_process)
        self._survival(X, Y, self.pending_mutation_types)
        self.tell_count += 1

        self._update_archive()

        # with open('test_log.txt', 'a') as f:
        #     f.write(f'Archive: {self.archive.individuals},'
        #             '\n'
        #             f'Fitness: {self.archive.fitnesses},'
        #             '\n'
        #             f'Descriptor: {self.archive.descriptors}'
        #             '\n'
        #             f'Length: {len(self.archive)}'
        #             '\n\n')
