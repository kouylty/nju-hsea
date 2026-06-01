# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import os
import numpy as np
import pandas as pd
from benchmarks import EarthBenchmark_278
from algorithms._utils import kendall_ranking_correlation

class EarthBenchEnv(gym.Env):
    def __init__(self, p1, p2, dims, f1=None, f2=None, constraints_dict=None, max_steps=None):
        self.dims = dims
        self.obs_dim = dims * 3
        self.action_dim = 3 # 只有 3 个动作，即从父代1的边中选择，从父代2的边中选择，从所有边中随机选择一个
        self.max_steps = max_steps if max_steps is not None else dims  # 默认最大步数 = 维度

        # 需要保证两个父代都是合法的
        eb = EarthBenchmark_278()
        eb.repair(p1)
        eb.repair(p2)

        self.p1 = p1
        self.p2 = p2
        self.f1 = f1 if f1 is not None else eb(p1)
        self.f2 = f2 if f2 is not None else eb(p2)
        self.best_f = max(self.f1, self.f2)
        self.best_p = p1 if self.f1 > self.f2 else p2

        self.observation_space = spaces.Box(low=0, high=self.dims-1, shape=(self.obs_dim,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.action_dim)

        self.random_seed = 0

        # 避免重复加载
        if constraints_dict == None:
            self.load_constraints()
        else:
            self.constraints_dict = constraints_dict

    def reset(self, seed=None):
        super().reset(seed=seed if seed is not None else self.random_seed)
        # 选择 self.best_p 的第一个点作为初始点
        self.o_partial = [self.best_p[0]]
        self.step_count = 1
        return self._get_obs(), {}

    def step(self, action):
        # 根据动作选择下一个元素
        if action == 0:
            chosen, reward = self._choose_from(self.p1)
        elif action == 1:
            chosen, reward = self._choose_from(self.p2)
        else:
            chosen, reward = self._choose_random()

        if chosen is None:
            return self._get_obs(), reward, False, False, {} # (obs, reward, terminated, truncated, info)
        self.o_partial.append(chosen)
        self.step_count += 1

        # done = (self.step_count >= self.dims) # old version
        terminated = (self.step_count >= self.dims)
        truncated = (self.step_count >= self.max_steps)
        if terminated:
            reward = self._calculate_final_reward()

        return self._get_obs(), reward, terminated, truncated, {} # (obs, reward, terminated, truncated, info)

    def _calculate_final_reward(self):
        fitness = EarthBenchmark_278()(self.o_partial)
        # reward = kendall_ranking_correlation(self.p2, self.o_partial, device='cpu').item() * 500
        # print(f"Final fitness: {fitness}, f1: {self.f1}, f2: {self.f2}, best_f: {self.best_f}, better? {fitness > self.best_f}")
        return (fitness - self.best_f)

    def _choose_from(self, parent):
        cur = self._get_cur_node()
        candidates = []

        # 找到当前点在 parent 中的位置
        idx = np.where(parent == cur)[0]
        assert len(idx) != 0 # 不在 parent 中
        idx = idx[0]

        # 相邻点
        # if idx > 0:
        #     candidates.append(parent[idx - 1])
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
            chosen, _ = self._choose_random()
            return chosen, 0.0
        else:
            # 随机选择一个合法候选
            chosen = np.random.choice(valid_candidates)
            return chosen, 0.0
        
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
        return np.random.choice(valid), 0.0

    def _get_obs(self):
        pad_len = self.dims - len(self.o_partial)
        o_vec = self.o_partial + [0] * pad_len
        obs = np.concatenate([self.p1, self.p2, np.array(o_vec, dtype=np.int32)])
        return obs.astype(np.int32)

    def _get_cur_node(self):
        return self.o_partial[-1]

    # def seed(self, seed):
    #     self.random_seed = seed

    def load_constraints(self):
        constraints_path = 'Environments/CONOPLib_278/constraints.npy'
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
