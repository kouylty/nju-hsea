import numpy as np
from typing import List
import logging
import random
from ._base import BaseOptimizer
from ._utils import get_init_samples
from ._ea_operator import (
    swap_mutation, insert_mutation, reversal_mutation, shuffle_mutation, shift_mutation,
    order_crossover, pmx_crossover, cycle_crossover
)

log = logging.getLogger(__name__)


class BanditManager:
    def __init__(self, operators, algo="ucb", epsilon=0.1):
        self.operators = operators
        self.n_ops = len(operators)
        self.counts = np.zeros(self.n_ops)   # 拉动次数
        self.values = np.zeros(self.n_ops)   # 平均奖励
        self.algo = algo
        self.epsilon = epsilon
        self.t = 0  # 总轮数

    def select_operator(self):
        self.t += 1
        if self.algo == "epsilon_greedy":
            if random.random() < self.epsilon:
                return random.choice(self.operators)
            else:
                return self.operators[np.argmax(self.values)]
        elif self.algo == "ucb":
            # # 每 200 * 20 次重置
            # if self.t % 4000 == 0:
            #     self.t = 0
            #     self.counts = np.zeros(self.n_ops)
            #     self.values = np.zeros(self.n_ops)
            ucb_values = self.values + np.sqrt(2*np.log(self.t) / (self.counts + 1e-9))
            # print(f"ucb_values: {ucb_values}, self.t: {self.t}")
            return self.operators[np.argmax(ucb_values)]
        else:
            raise NotImplementedError

    def update(self, op, reward):
        idx = self.operators.index(op)
        self.counts[idx] += 1
        # 增量平均
        self.values[idx] += (reward - self.values[idx]) / self.counts[idx]

    # def update(self, op, reward, alpha=0.1):
    #     idx = self.operators.index(op)
    #     self.counts[idx] += 1
    #     self.values[idx] = (1 - alpha) * self.values[idx] + alpha * reward


class BanditEA(BaseOptimizer):
    def __init__(
        self, dims, lb, ub, pop_size=20, init_sampler_type='permutation',
        selection_type='elite', bandit_algo="ucb", epsilon=0.1
    ):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.offspring_size = self.pop_size
        self.init_sampler_type = init_sampler_type
        self.selection_type = selection_type

        # population
        self.population = []
        self.fitness = []

        # Bandit 管理器
        self.mutation_bandit = BanditManager(
            operators=["swap", "insert", "reversal", "shuffle", "shift"],
            algo=bandit_algo, epsilon=epsilon
        )
        self.crossover_bandit = BanditManager(
            operators=["order", "pmx", "cycle"],
            algo=bandit_algo, epsilon=epsilon
        )

        # 存储最近一次使用的算子，便于 tell 更新奖励
        self._last_ops = []

    def _init_samples(self, init_sampler_type, n) -> List[np.ndarray]:
        points = get_init_samples(init_sampler_type, n, self.dims, self.lb, self.ub)
        return [points[i] for i in range(len(points))]

    def _mutation(self, x, op, n_repeats=1):
        if op == 'swap':
            next_x = swap_mutation(x, repeats=n_repeats)
        elif op == 'insert':
            next_x = insert_mutation(x, repeats=n_repeats)
        elif op == 'reversal':
            next_x = reversal_mutation(x, repeats=n_repeats)
        elif op == 'shuffle':
            next_x = shuffle_mutation(x, repeats=n_repeats)
        elif op == 'shift':
            next_x = shift_mutation(x)
        else:
            raise NotImplementedError
        return next_x

    def _crossover(self, x1, x2, op):
        if op == 'order':
            next_x = order_crossover(x1, x2)
        elif op == 'pmx':
            next_x = pmx_crossover(x1, x2)
        elif op == 'cycle':
            next_x = cycle_crossover(x1, x2)
        else:
            raise NotImplementedError
        return next_x

    def _apply_ops(self, x1, x2, cx_op, mt_op):
        next_x = self._crossover(x1, x2, cx_op)
        next_x = self._mutation(next_x, mt_op)
        return next_x

    def _selection(self, parents, parents_fit, offspring, offspring_fit, kappa=0.001):
        if self.selection_type == 'elite':
            all_individual = parents + offspring
            all_fitness = parents_fit + offspring_fit
            indices = np.argsort(all_fitness)[-self.pop_size:]
            next_generation = [all_individual[idx] for idx in indices]
            next_fitness = [all_fitness[idx] for idx in indices]
            return next_generation, next_fitness

        elif self.selection_type == 'rank_based_prioritized':
            all_individual = parents + offspring
            all_fitness = parents_fit + offspring_fit
            N = len(all_individual)
            sorted_indices = np.argsort(all_fitness)[::-1]
            ranks = np.empty(N, dtype=int)
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank
            probs = 1.0 / (kappa * N + ranks)
            probs /= probs.sum()
            indices = np.random.choice(N, size=self.pop_size, replace=False, p=probs)
            next_generation = [all_individual[i] for i in indices]
            next_fitness = [all_fitness[i] for i in indices]
            return next_generation, next_fitness
        else:
            raise NotImplementedError

    def ask(self, n_repeats=1):
        offspring = []
        self._last_ops = []  # 清空记录
        if len(self.population) == 0:
            offspring.extend(self._init_samples(self.init_sampler_type, self.pop_size))
        else:
            for _ in range(self.offspring_size):
                i, j = np.random.choice(range(self.pop_size), 2, replace=False)
                x1, x2 = self.population[i], self.population[j]
                cx_op = self.crossover_bandit.select_operator()
                mt_op = self.mutation_bandit.select_operator()
                next_x = self._apply_ops(x1, x2, cx_op, mt_op)
                offspring.append(next_x)
                self._last_ops.append((cx_op, mt_op))
            # print(f"self._last_ops: {self._last_ops}, len: {len(self._last_ops)}")
        return offspring

    # def tell(self, X: List[np.ndarray], Y: List):
    #     if len(self.population) == 0:
    #         self.population, self.fitness = X, Y
    #     else:
    #         parent_max_fitness = np.max(self.fitness)
    #         # parent_mean_fitness = np.mean(self.fitness)
    #         for i, (cx_op, mt_op), y in enumerate(zip(self._last_ops, Y)):
    #             # reward = max(0, y - parent_max_fitness)
    #             reward = np.clip(y - parent_max_fitness, -10, 10000)
    #             self.crossover_bandit.update(cx_op, reward)
    #             self.mutation_bandit.update(mt_op, reward)
    #         self.population, self.fitness = self._selection(self.population, self.fitness, X, Y)
    def tell(self, X: List[np.ndarray], Y: List):
        if len(self.population) == 0:
            self.population, self.fitness = X, Y
        else:
            parent_max_fitness = np.max(self.fitness)
            
            best_offspring_idx = np.argmax(Y)
            best_y = Y[best_offspring_idx]
            cx_op, mt_op = self._last_ops[best_offspring_idx]
            # reward = np.clip(best_y - parent_max_fitness, -10, 10000)
            reward = np.clip(best_y - parent_max_fitness, -5, 100000)

            self.crossover_bandit.update(cx_op, reward)
            self.mutation_bandit.update(mt_op, reward)
                
            self.population, self.fitness = self._selection(self.population, self.fitness, X, Y)