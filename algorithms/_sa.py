import numpy as np
from typing import List
import logging
import random
from ._base import BaseOptimizer
from ._ea_operator import swap_mutation, insert_mutation, reversal_mutation, shuffle_mutation, shift_mutation
from ._utils import get_init_samples

log = logging.getLogger(__name__)


class SA(BaseOptimizer):
    def __init__(self, dims, lb, ub, decay, T, update_freq, mutation_type='swap', init_sampler_type='permutation'):
        self.dims = dims
        self.decay = decay
        self.init_T = T
        self.T = T
        self.update_freq = update_freq
        self.mutation_type = mutation_type
        self.init_sampler_type = init_sampler_type
        self.lb = lb
        self.ub = ub

        self.best_x = None
        self.best_y = None
        self.cnt = 0

    def _mutation(self, x, n_repeats=1):
        if self.mutation_type == 'mix':
            # 定义每个mutation类型及其对应的概率
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

    def ask(self, n_repeats=1) -> List[np.ndarray]:
        if self.best_x is None:
            x = get_init_samples(self.init_sampler_type, 1, self.dims, self.lb, self.ub)[0]
        else:
            x = self._mutation(self.best_x, n_repeats=1)
        return [x]

    def tell(self, X: List[np.ndarray], Y: List):
        if self.best_x is None:
            self.best_x = X[0]
            self.best_y = Y[0]

        # simulated annealing
        for x, y in zip(X, Y):
            if y > self.best_y:
                self.best_x = x 
                self.best_y = y 
            else:
                probability = np.exp(-(self.best_y - y) / self.T)
                if np.random.uniform(0, 1) < probability:
                    self.best_x = x
                    self.best_y = y 

        self.cnt += 1
        if self.cnt % self.update_freq == 0:
            self.cnt = 0
            self.T = self.decay * self.T