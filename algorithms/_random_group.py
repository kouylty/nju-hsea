import numpy as np
from typing import List
import logging
from ._base import BaseOptimizer
from algorithms import BO, SA, EA, Random
from ._ea_operator import swap_mutation
from ._utils import get_init_samples, select, get_subset

log = logging.getLogger(__name__)


class RandomGroup(BaseOptimizer):
    def __init__(self, dims, lb, ub, n_init, active_dims, **config):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.n_init = n_init
        self.active_dims = active_dims
        
        self.train_X = []
        self.train_Y = []
        self.best_x = None
        self.best_y = None

    def _get_random_group(self, dims, active_dims):
        perm = np.random.permutation(dims)
        group = []
        for i in range(int(np.ceil(dims/active_dims))):
            group.append(perm[i*active_dims: (i+1)*active_dims])
        print('perm:', perm)
        print('group')
        print(group)
        return group
    
    def optimize(self):
        group = self._get_random_group(self.dims, self.active_dims)
        next_x = np.zeros(self.dims)
        for idx in group:
            tmp_x = swap_mutation(self.best_x[idx])
            next_x[idx] = tmp_x
        assert len(set(next_x)) == self.dims
        return [next_x]

    def ask(self, n_repeats=1) -> List[np.ndarray]:
        if len(self.train_X) == 0:
            new_X = get_init_samples('permutation', self.n_init, self.dims, self.lb, self.ub)
            new_X = [new_X[i] for i in range(self.n_init)]
        else:
            new_X = self.optimize()
        
        return new_X

    def tell(self, X: List[np.ndarray], Y: List):
        self.train_X.extend(X)
        self.train_Y.extend(Y)
        for x, y in zip(X, Y):
            if self.best_y is None or self.best_y < y:
                self.best_x = x
                self.best_y = y
