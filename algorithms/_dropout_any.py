import numpy as np
from typing import List
import logging
from ._base import BaseOptimizer
from algorithms import BO, SA, EA, Random
from ._fillin_strategy import PermutationRandomStrategy, PermutationBestKPosStrategy, PermutationBestKOrderStrategy
from ._utils import get_init_samples, select, get_subset

log = logging.getLogger(__name__)


class DropoutAny(BaseOptimizer):
    def __init__(self, dims, lb, ub, n_init, active_dims, inner_opt_type, fillin_type, reset_freq, **config):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.n_init = n_init
        self.active_dims = active_dims
        self.inner_opt_type = inner_opt_type
        self.reset_freq = reset_freq
        self.config = config

        fillin_strategy_factory = {
            'random': PermutationRandomStrategy(dims, lb, ub),
            'best_pos': PermutationBestKPosStrategy(dims, lb, ub, 10),
            'best_order': PermutationBestKOrderStrategy(dims, lb, ub, 10),
        }
        self.fillin_strategy = fillin_strategy_factory[fillin_type]

        # def map_fn(k, map_dict):
        #     return map_dict[k]
        # self.map_fn = np.vectorize(map_fn)

        self.train_X = []
        self.train_Y = []

        self._reset_inner_opt()

    def _select(self, dims, active_dims):
        idx = select(dims, active_dims)
        return idx

    def _get_inner_opt(self, inner_opt_type):
        inner_opt_class = {
            'bo': BO,
            'sa': SA,
            'ea': EA,
            'random': Random,
        }
        inner_opt = inner_opt_class[inner_opt_type](dims=self.dims, lb=self.lb, ub=self.ub, **self.config)
        return inner_opt

    def ask(self, n_repeats=1) -> List[np.ndarray]:
        if len(self.train_X) == 0:
            new_X = get_init_samples('permutation', self.n_init, self.dims, self.lb, self.ub)
            new_X = [new_X[i] for i in range(self.n_init)]
        else:
            proposed_X = self.inner_opt.ask()

            # fill in
            new_X = []
            for i in range(len(proposed_X)):
                fixed_vars = {j: pos for j, pos in zip(self.idx, proposed_X[i])}
                new_x = self.fillin_strategy.fillin(fixed_vars)
                new_X.append(new_x)
        
        return new_X

    def tell(self, X: List[np.ndarray], Y: List):
        self.train_X.extend(X)
        self.train_Y.extend(Y)
        self._subset_tell(X, Y, self.inner_opt)

        for x, y in zip(X, Y):
            self.fillin_strategy.update(x.reshape(1, -1), y)

        self.cnt += 1
        if self.cnt % self.reset_freq == 0:
            self._reset_inner_opt()

    def _subset_tell(self, X, Y, opt):
        X_np = np.vstack(X)
        subset_X = get_subset(X_np, self.idx)
        subset_X = [subset_X[i] for i in range(len(subset_X))]
        opt.tell(subset_X, Y)

    def _reset_inner_opt(self):
        log.info('reset inner optimizer')
        self.cnt = 0
        self.idx = self._select(self.dims, self.active_dims)
        # self.idx_map = dict(zip(self.idx, range(self.active_dims)))
        # self.idx_inv_map = dict(zip(range(self.active_dims), self.idx))

        self.inner_opt = self._get_inner_opt(self.inner_opt_type)

        if len(self.train_X) > 0:
            self._subset_tell(self.train_X, self.train_Y, self.inner_opt)
