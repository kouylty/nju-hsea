# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms._utils import kendall_ranking_correlation, segmented_kendall_correlation


class EarthBenchEnvUnified(gym.Env):
    def __init__(self, p1, p2, reference, dims, f1=None, f2=None, num_segments=None, constraints_dict=None, is_test=False, local_search=False, parent_local_search=False, device='cpu'):
        self.dims = dims
        self.num_segments = num_segments
        if num_segments is None:
            self.obs_dim = 9 # 统一为 9 维观测空间，包含 (f1, f2, kendall1, kendall2, l, u1, u2, v1, v2)
        else:
            self.obs_dim = 9 + 2 * num_segments
        self.action_dim = 3 # 只有 3 个动作，即从父代1的边中选择，从父代2的边中选择，从所有边中随机选择一个
        self.a1_counter = 0
        self.a2_counter = 0
        self.a3_counter = 0
        self.a1_fail_counter = 0
        self.a2_fail_counter = 0

        self.is_test = is_test
        
        if dims == 124:
            from benchmarks import EarthBenchmark_124 as EarthBenchmark
            self.f_min, self.f_max = -5_000, 0 # -8000
            # self.f_min, self.f_max = -4_200, 0 # -8000
        elif dims == 278:
            from benchmarks import EarthBenchmark_278 as EarthBenchmark
            self.f_min, self.f_max = -3_500, 0 # -8000
            # self.f_min, self.f_max = -2_400, 0 # -8000
        elif dims == 902:
            from benchmarks import EarthBenchmark_902 as EarthBenchmark
            self.f_min, self.f_max = -3_000_000, 0
        elif dims == 904:
            from benchmarks import EarthBenchmark_904 as EarthBenchmark
            self.f_min, self.f_max = -3_000_000, 0
        elif dims == 934:
            from benchmarks import EarthBenchmark_934 as EarthBenchmark
            self.f_min, self.f_max = -15_000_000, 0
        elif dims == 2538:
            from benchmarks import EarthBenchmark_2538 as EarthBenchmark
            self.f_min, self.f_max = -2_000_000, 0
        elif dims == 2574:
            from benchmarks import EarthBenchmark_2574 as EarthBenchmark
            self.f_min, self.f_max = -7_500_000, 0
        elif dims == 2582:
            from benchmarks import EarthBenchmark_2582 as EarthBenchmark
            self.f_min, self.f_max = -12_000_000, 0
        else:
            raise ValueError(f"Unsupported dims: {dims}")
        
        # 需要保证两个父代都是合法的
        self.eb = EarthBenchmark()
        self.eb.repair(p1)
        self.eb.repair(p2)
        self.eb.repair(reference)

        self.p1 = p1
        self.p2 = p2
        self.ref = reference
        self.f1 = f1 if f1 is not None else self.eb(p1)
        self.f2 = f2 if f2 is not None else self.eb(p2)

        self.kendall1 = kendall_ranking_correlation(self.ref, self.p1, device=device).item()
        self.kendall2 = kendall_ranking_correlation(self.ref, self.p2, device=device).item()

        if self.num_segments is not None:
            self.kendall1_vec = segmented_kendall_correlation(self.p1, self.ref, num_segments=self.num_segments, device=device)
            self.kendall2_vec = segmented_kendall_correlation(self.p2, self.ref, num_segments=self.num_segments, device=device)
        # print(f"Kendall correlation, p1: {self.kendall1}, p2: {self.kendall2}")

        # local search for parents
        self.local_search = local_search
        if parent_local_search:
            print(f"Before local search, p1: {self.f1}, p2: {self.f2}")
            self.p1, self.f1 = self._local_search(self.p1, self.f1, epochs=200)
            self.p2, self.f2 = self._local_search(self.p2, self.f2, epochs=200)
            print(f"After local search, p1: {self.f1}, p2: {self.f2}")

        # print(f"Fitness, p1: {self.f1}, p2: {self.f2}")
        self.target = max(self.f1, self.f2)
        self.best_p = p1 if self.f1 > self.f2 else p2

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_dim)

        self.random_seed = 0

        # 避免重复加载
        if constraints_dict == None:
            self.load_constraints()
        else:
            self.constraints_dict = constraints_dict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self.random_seed)
        # 选择 self.best_p 的第一个点作为初始点
        self.o_partial = [self.best_p[0]]
        self.step_count = 1
        return self._get_obs(), {}

    def step(self, action):
        # 根据动作选择下一个元素
        if action == 0:
            self.a1_counter += 1
            chosen, reward = self._choose_from(self.p1)
            if reward < 0:
                self.a1_fail_counter += 1
        elif action == 1:
            self.a2_counter += 1
            chosen, reward = self._choose_from(self.p2)
            if reward < 0:
                self.a2_fail_counter += 1
        else:
            self.a3_counter += 1
            chosen, reward = self._choose_random()

        if chosen is None:
            return self._get_obs(), reward, False, False, {} # (obs, reward, terminated, truncated, info)
        self.o_partial.append(chosen)
        self.step_count += 1

        # done = (self.step_count >= self.dims) # old version
        terminated = (self.step_count >= self.dims)
        truncated = (self.step_count >= self.dims)
        if terminated:
            reward = self._calculate_final_reward()
            

        return self._get_obs(), reward, terminated, truncated, {} # (obs, reward, terminated, truncated, info)

    def _calculate_final_reward(self):
        fitness = self.eb(self.o_partial)
        if self.local_search:
            _, fitness = self._local_search(self.o_partial, fitness, epochs=200)
        # reward = kendall_ranking_correlation(self.p2, self.o_partial, device='cpu').item() * 500
        # print(f"Final fitness: {fitness}, f1: {self.f1}, f2: {self.f2}, best_f: {self.best_f}, better? {fitness > self.best_f}")
        return (fitness - self.target)

    def _choose_from(self, parent):
        cur = self._get_cur_node()
        if cur is None:
            candidates = list(range(0, self.dims, 2))
        else:
            candidates = []

            # 找到当前点在 parent 中的位置
            idx = np.where(parent == cur)[0]
            assert len(idx) != 0 # 不在 parent 中
            idx = idx[0]

            # 相邻点
            if idx > 0:
                candidates.append(parent[idx - 1])
            if idx < len(parent) - 1:
                candidates.append(parent[idx + 1])

        # --- 约束处理 ---
        # 1. 去掉已经出现过的
        candidates = [c for c in candidates if c not in self.o_partial]

        # 2. constraint 约束检查
        valid_candidates = []
        for c in candidates:
            if c in self.constraints_dict:
                required = self.constraints_dict[c]
                # 如果 required 的任一元素没出现，就不能选
                if not required.issubset(self.o_partial):
                    continue
            valid_candidates.append(c)

        # 如果没有合法候选，就 fallback：从所有未出现的点中随机选一个
        if not valid_candidates:
            # 没有合法候选，返回 None 表示失败，奖励为 -10
            chosen, _ = self._choose_random()
            if self.is_test:
                return chosen, 0.0
            else:
                return chosen, -5.0
        else:
            # 随机选择一个合法候选
            chosen = np.random.choice(valid_candidates)
            if self.is_test:
                return chosen, 0.0
            else:
                return chosen, 1.0 # 成功，奖励为 1.0
        
    def _choose_random(self):
        # 1. 所有未出现的点
        all_nodes = set(range(self.dims))
        remaining = list(all_nodes - set(self.o_partial))

        # 2. 检查约束
        valid = []
        for c in remaining:
            if c in self.constraints_dict:
                required = self.constraints_dict[c]
                # 如果 required 中有点还没出现，则不能选
                if not required.issubset(self.o_partial):
                    continue
            valid.append(c)

        # 3. 随机选择
        if not valid:
            assert False  # 理论上不该出现
            return None, -50.0  
        return np.random.choice(valid), 0

    def is_valid(self, node):
        if node in self.o_partial:
            return -1
        if node in self.constraints_dict:
            required = self.constraints_dict[node]
            if not required.issubset(self.o_partial):
                return -1
    
    def _get_candidates(self):
        cur = self._get_cur_node()
        candidates = []

        # 找到当前点在 parent 中的位置
        idx = np.where(self.p1 == cur)[0]
        assert len(idx) != 0 # 不在 parent 中
        idx = idx[0]

        # 相邻点
        if idx > 0:
            candidates.append(self.p1[idx - 1])
        else:
            candidates.append(-1) # 前面没有，用 -1 填充，表示无效
        if idx < len(self.p1) - 1:
            candidates.append(self.p1[idx + 1])
        else:
            candidates.append(-1) # 后面没有，用 -1 填充，表示无效
        
        idx = np.where(self.p2 == cur)[0]
        assert len(idx) != 0 # 不在 parent 中
        idx = idx[0]

        if idx > 0:
            candidates.append(self.p2[idx - 1])
        else:
            candidates.append(-1) 
        if idx < len(self.p2) - 1:
            candidates.append(self.p2[idx + 1])
        else:
            candidates.append(-1)

        return candidates

    def cal_descriptor(self, cur, candidate):
        if candidate == -1:
            return 0.0
        # 在 self.ref 中找到 cur 和 candidate 的位置
        idx_cur = np.where(self.ref == cur)[0][0]
        idx_cand = np.where(self.ref == candidate)[0][0]

        sign = 1 if idx_cand > idx_cur else -1
        return sign * abs(idx_cand - idx_cur) / self.dims

    def _get_obs(self):
        cur = self._get_cur_node()
        
        candidates = self._get_candidates()
        filtered_candidates = [self.is_valid(c) for c in candidates] # -1 表示无效，但是保留

        pi1_pi2 = [self.cal_descriptor(cur, c) for c in candidates]

        f1_norm = (self.f1 - self.f_min) / (self.f_max - self.f_min)
        f2_norm = (self.f2 - self.f_min) / (self.f_max - self.f_min)
        if self.num_segments is not None:
            obs = np.concatenate([[f1_norm, f2_norm, self.kendall1, self.kendall2, self.step_count / self.dims],
                                  np.array(pi1_pi2, dtype=np.float32),
                                  np.array(self.kendall1_vec, dtype=np.float32),
                                  np.array(self.kendall2_vec, dtype=np.float32)])
        else:
            obs = np.concatenate([[f1_norm, f2_norm, self.kendall1, self.kendall2, self.step_count / self.dims],
                                np.array(pi1_pi2, dtype=np.float32)])
        return obs

    def _get_cur_node(self):
        if self.step_count == 0:
            return None
        else:
            return self.o_partial[-1]

    # def seed(self, seed):
    #     self.random_seed = seed

    def _local_search(self, x, y, epochs=200):
        from utils import local_search
        best_x, best_y = local_search(x, y, dims=self.dims, epochs=epochs)
        return best_x, best_y

    def load_constraints(self):
        constraints_path = f'Environments/constraints_{self.dims}.npy'
        if os.path.exists(constraints_path):
            self.constraints = np.load(constraints_path)
            
            # 处理成字典形式，提高查询效率
            self.constraints_dict = {}
            for a, b in self.constraints:
                if b not in self.constraints_dict:
                    self.constraints_dict[b] = set()
                self.constraints_dict[b].add(a)
            return

        bench_dir = f'benchmarks/CONOPLib_{self.dims}'
        df = pd.read_csv(f'{bench_dir}/coex.dat', delim_whitespace=True, header=None)
        coex = df.loc[:].values[:, :2]

        df = pd.read_csv(f'{bench_dir}/Fb4L.dat', delim_whitespace=True, header=None)
        FadLad = df.loc[:].values

        from algorithms._dqn_utils import coex2constraints, FADLAD2constraints
        coex_constraints = coex2constraints(coex)
        FadLad_constraints = FADLAD2constraints(FadLad)

        constraints = np.vstack([coex_constraints, FadLad_constraints])
        # 去除重复项，经检查去除重复项后已经是闭包，后面的求解闭包可以省略
        self.constraints = np.unique(constraints, axis=0)
        np.save(constraints_path, self.constraints)
        print("Saved constraints to file, shape:", self.constraints.shape)

        # 处理成字典形式，提高查询效率
        self.constraints_dict = {}
        for a, b in self.constraints:
            if b not in self.constraints_dict:
                self.constraints_dict[b] = set()
            self.constraints_dict[b].add(a)
        
        # from algorithms._dqn_utils import compute_closure_optimized
        # closure = compute_closure_optimized(constraints)
        # print(f"传递闭包之后闭包的shape: {closure.shape}")

    def _get_constraints_dict(self):
        return self.constraints_dict


if __name__ == "__main__":
    np.random.seed(42)
    p1 = np.random.permutation(124)
    p2 = np.random.permutation(124)
    ref = np.random.permutation(124)

    env = EarthBenchEnvUnified(p1, p2, ref, dims=124, num_segments=16)
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs[:9])
        print(obs[9:9+16])
        print(obs[9+16:])
        break
        done = terminated or truncated
        total_reward += reward
    print("Total Reward:", total_reward)