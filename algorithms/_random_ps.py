import numpy as np
from typing import List
import logging
from ._base import BaseOptimizer
from ._fillin_strategy import PermutationRandomStrategy, PermutationBestKPosStrategy
from ._utils import get_init_samples, permutation_sampler, select, get_subset

log = logging.getLogger(__name__)


class Random(BaseOptimizer):
    def __init__(self, dims, lb, ub, active_dims, fillin_type='random'):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.active_dims = active_dims

        fillin_strategy_factory = {
            'random': PermutationRandomStrategy(dims, lb, ub),
            'best_pos': PermutationBestKPosStrategy(dims, lb, ub, 10),
        }
        self.fillin_strategy = fillin_strategy_factory[fillin_type]

        self.best_x = None
        self.best_y = None

    def ask(self, n_repeats=1) -> List[np.ndarray]:
        if self.best_x is None:
            x = get_init_samples('permutation', 1, self.dims, self.lb, self.ub)[0]
        else:
            idx = select(self.dims, self.active_dims)
            # generate random position
            proposed_X = np.random.choice(range(self.dims), self.active_dims, replace=False)
            fixed_vars = {j: pos for j, pos in zip(idx, proposed_X)}
            x = self.fillin_strategy.fillin(fixed_vars)

        return [x]
    
    def tell(self, X: List[np.ndarray], Y: List):
        assert len(Y) == 1
        x, y = X[0], Y[0]
        if self.best_y is None or y > self.best_y:
            self.best_x = x
            self.best_y = y
        self.fillin_strategy.update(x.reshape(1, -1), y)