import torch
from torch import Tensor
from multiprocessing import Pool
import gpytorch
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP
from botorch.acquisition import ExpectedImprovement
import numpy as np
from collections import deque
from collections import OrderedDict
from collections import Counter
from typing import List
import time
import logging
from ._ea import EA
from ._ea_operator import swap_mutation
from ._base import BaseOptimizer
from ._fillin_strategy import PermutationRandomStrategy, PermutationBestKPosStrategy
from ._utils import get_init_samples, permutation_sampler, select, get_subset, featurize

log = logging.getLogger(__name__)

class FeatureCache:
    def __init__(self, max_size=500):
        self.cache = OrderedDict()
        self.max_size = max_size  # 设置缓存最大大小

    def _get_key(self, x):
        if x.dim() == 1:  # 单个输入
            return tuple(x.tolist())
        elif x.dim() == 2:  # 批量输入
            return [tuple(x[i].tolist()) for i in range(x.size(0))]

    def _evict_if_needed(self):
        # 如果缓存大小超过限制，移除最早的项
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def push(self, x):
        feature = self.get(x)
        if feature is None:
            feature = featurize(x, 'torch')
            self.cache[self._get_key(x)] = feature
            self._evict_if_needed()  # 检查并清理缓存
        return feature

    def get(self, x):
        key = self._get_key(x)
        if isinstance(key, list):  # 批量情况
            cached_features = [self.cache[k] for k in key if k in self.cache]
            return torch.stack(cached_features) if cached_features else None
        return self.cache.get(key, None)

    def push_batch(self, X):
        keys = self._get_key(X)
        key_counts = Counter(keys)  # 记录每个键出现的次数
        
        # 已缓存特征和未缓存特征索引
        cached_features = []
        new_indices = []
        new_keys = []
        
        for i, key in enumerate(keys):
            if key in self.cache:
                cached_features.append(self.cache[key])
            else:
                new_indices.append(i)
                new_keys.append(key)
        
        # 计算新特征
        if new_keys:
            new_features = featurize(X[new_indices], 'torch')  # 批量计算新特征
            for key, feature in zip(new_keys, new_features):
                self.cache[key] = feature
            cached_features.extend(new_features)

        # 根据 keys 顺序组织返回特征
        result_features = [self.cache[key] for key in keys]
        self._evict_if_needed()  # 检查并清理缓存
        return torch.stack(result_features)

class BO(BaseOptimizer):
    def __init__(
        self, dims, lb, ub, active_dims=None, n_init=10, batch_size=1, init_sampler_type='permutation', 
        acqf_init_sampler_type='permutation', acqf_type='EI', acqf_opt_type='random', kernel_type='rbf', 
        fillin_type='random', device='cpu'
    ):
        self.dims = dims
        self.active_dims = active_dims
        self.lb = np.ones(self.dims) * lb
        self.ub = np.ones(self.dims) * ub
        self.n_init = n_init
        self.batch_size = batch_size
        self.init_sampler_type = init_sampler_type
        self.acqf_init_sampler_type = acqf_init_sampler_type
        self.acqf_type = acqf_type
        self.acqf_opt_type = acqf_opt_type
        self.kernel_type = kernel_type
        fillin_strategy_factory = {
            'random': PermutationRandomStrategy(dims, lb, ub),
            'best_pos': PermutationBestKPosStrategy(dims, lb, ub, 10),
        }
        self.fillin_strategy = fillin_strategy_factory[fillin_type]
        self.device = torch.device(device)
        if device == 'cuda' and not torch.cuda.is_available():
            log.warning("CUDA not available, using CPU")
            self.device = torch.device('cpu')
        log.info('Device: {}'.format(self.device))

        self.train_X = []
        self.train_Y = []
        self.cache = FeatureCache()

        self.cache_X = deque()

    def get_ckpt_dict(self):
        return {
            'train_X': self.train_X,
            'train_Y': self.train_Y, 
            'cache_X': list(self.cache_X)
        }

    def load_ckpt_dict(self, ckpt_dict):
        self.train_X = ckpt_dict['train_X']
        self.train_Y = ckpt_dict['train_Y']
        self.cache_X = deque(ckpt_dict['cache_X'])
        best_idx = np.argmax(self.train_Y)
        best_x = self.train_X[best_idx]
        best_y = self.train_Y[best_idx]
        return best_x, best_y
        
    def _init_samples(self, init_sampler_type, n) -> List[np.ndarray]:
        points = get_init_samples(init_sampler_type, n, self.dims, self.lb, self.ub)
        points = [points[i] for i in range(len(points))]
        return points
        
    def _get_kernel(self, kernel_type):
        if kernel_type == 'rbf':
            kernel = RBFKernel()
        elif kernel_type == 'matern':
            kernel = MaternKernel()
        else:
            raise NotImplementedError
        return kernel
        
    def _get_acqf(self, acqf_type, model, train_X: Tensor, train_Y: Tensor):
        if acqf_type == 'EI':
            AF = ExpectedImprovement(model, best_f=train_Y.max().item()).to(self.device)
        else:
            raise NotImplementedError
        return AF
        
    def _init_model(self, train_X: Tensor, train_Y: Tensor):
        Y_var = torch.full_like(train_Y, 0.01).to(self.device)
        kernel = self._get_kernel(self.kernel_type).to(self.device)
        model = FixedNoiseGP(train_X, train_Y, Y_var, covar_module=kernel).to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)
        return mll, model
    
    def _optimize_acqf_random(self, dims, AF, lb, ub, n=1):
        cand_X = permutation_sampler(1024, dims, list(range(self.dims)))
        cand_X = torch.from_numpy(cand_X).float().to(self.device)
        cand_Y = torch.cat([AF(X_) for X_ in cand_X.split(1)]).reshape(-1)
        # cand_Y = AF(cand_X.unsqueeze(1))
        indices = torch.argsort(cand_Y)[-n: ]
        proposed_X, proposed_Y = cand_X[indices], cand_Y[indices]
        return proposed_X, proposed_Y

    def _optimize_acqf_local_search(self, dims, AF, lb, ub, n=1, n_restart = 10):
        assert n == 1
        init_X, init_Y = self._optimize_acqf_random(dims, AF, lb, ub, n_restart)
        best_cand = None
        best_vals = None
        
        for i in range(n_restart):
            x = init_X[i].cpu()
            vals = init_Y[i].cpu()
            while True:
                all_cands = []
                all_vals = []
                # generate neighbors
                for i in range(dims):
                    for j in range(i+1, dims):
                        next_x = x.clone()
                        tmp = next_x[i].item()
                        next_x[i] = next_x[j].item()
                        next_x[j] = tmp
                        all_cands.append(next_x)
                        with torch.no_grad():
                            all_vals.append(AF(next_x.unsqueeze(0).to(self.device)))
                idx = torch.argmax(torch.cat(all_vals))
                if all_vals[idx] > vals:
                    x = all_cands[idx]
                    vals = all_vals[idx]
                else:
                    break
            if best_vals is None or vals > best_vals:
                best_cand = x
                best_vals = vals
            # log.debug('-----------------------')
            # log.debug('x: {}'.format(x))
            # log.debug('vals: {}'.format(vals))
            # log.debug('-----------------------')
            
        log.info('Local search finished!')
        return [best_cand], [best_vals]

    def _optimize_acqf_ea(self, dims, AF, lb, ub, n=1):
        # ea_alg = EA(dims, lb, ub, pop_size=18, init_sampler_type='permutation', mutation_type='swap', crossover_type='order')
        ea_alg = EA(dims, lb, ub, pop_size=10, init_sampler_type='permutation', mutation_type='insert', crossover_type='pmx')

        for _ in range(500):
            cands = ea_alg.ask()

            # 单个输入
            # cands_tensor = [torch.from_numpy(cand).float() for cand in cands]
            # cands_y_tensor = [AF(cand.unsqueeze(0)) for cand in cands_tensor]
            # cands_y = [y.cpu().detach().numpy().item() for y in cands_y_tensor]

            # 批量输入
            cands_tensor = torch.tensor(np.array(cands), dtype=torch.float32, device=self.device).unsqueeze(1)
            with torch.no_grad():
                cands_y_tensor = AF(cands_tensor)
            cands_y = cands_y_tensor.cpu().numpy().tolist()

            ea_alg.tell(cands, cands_y)

            # 删除分配的张量
            del cands_tensor, cands_y_tensor
            # torch.cuda.empty_cache()
        
        indices = np.argsort(ea_alg.fitness)[-n: ]
        proposed_X = [torch.from_numpy(ea_alg.population[idx]) for idx in indices]
        proposed_Y = [ea_alg.fitness[idx] for idx in indices]
        return proposed_X, proposed_Y

    def _optimize_acqf(self, dims, AF, lb, ub, n=1):
        if self.acqf_opt_type == 'random':
            proposed_X, proposed_Y = self._optimize_acqf_random(dims, AF, lb, ub, n)
        elif self.acqf_opt_type == 'ls':
            proposed_X, proposed_Y = self._optimize_acqf_local_search(dims, AF, lb, ub, n)
        elif self.acqf_opt_type == 'ea':
            proposed_X, proposed_Y = self._optimize_acqf_ea(dims, AF, lb, ub, n)
        else:
            raise NotImplementedError
        return proposed_X, proposed_Y
    
    def ask(self, n_repeats=1) -> List[np.ndarray]:
        # init
        if len(self.cache_X) + len(self.train_X) < self.n_init:
            points = self._init_samples(self.init_sampler_type, self.n_init)
            self.cache_X.extend(points)
            
        # unevaluated points
        if len(self.cache_X) > 0:
            return [self.cache_X.popleft()]
        
        # prepare train data
        train_X_tensor = torch.vstack(self.train_X).float().to(self.device)
        subset_X = None
        if self.active_dims is not None:
            idx = select(self.dims, self.active_dims)
            subset_X = get_subset(train_X_tensor, idx).to(self.device, dtype = torch.float64)
        
        train_Y_tensor = torch.from_numpy(np.vstack(self.train_Y)).to(self.device)
        train_Y_tensor = (train_Y_tensor - train_Y_tensor.mean()) / (train_Y_tensor.std() + 1e-6)

        # train model
        mll, model = self._init_model(train_X_tensor if subset_X is None else subset_X, train_Y_tensor)
        fit_gpytorch_model(mll)
        
        # optimize acquisition function
        AF = self._get_acqf(self.acqf_type, model, train_X_tensor if subset_X is None else subset_X, train_Y_tensor).to(self.device)
        proposed_X, _ = self._optimize_acqf(self.dims if self.active_dims is None else self.active_dims, AF, self.lb, self.ub, 1)
        # log.debug('Proposed X: {}'.format(proposed_X))
        assert len(proposed_X) == 1

        # # 一定概率修改得到的排列
        # if np.random.uniform(0, 1) < 0.9:
        #     proposed_X[:] = [proposed_X[i].cpu().detach().numpy() for i in range(len(proposed_X))]
        # else:
        #     proposed_X[:] = [swap_mutation(proposed_X[i].cpu().detach().numpy()) for i in range(len(proposed_X))]

        # log.info('Optimize acquisition function time: {}'.format(time.time() - st))

        if self.active_dims is not None:
            # fill in
            new_X = []
            for i in range(len(proposed_X)):
                fixed_vars = {j: pos for j, pos in zip(idx, proposed_X[i])}
                log.debug('fixed variables: {}'.format(fixed_vars))
                new_x = self.fillin_strategy.fillin(fixed_vars)
                log.debug('new x: {}'.format(new_x))
                new_X.append(new_x)
        
            self.cache_X.extend(new_X)
            return [self.cache_X.popleft()]
        else:
            self.cache_X.extend(proposed_X)
            return [self.cache_X.popleft()]
        
    
    def tell(self, X: List[np.ndarray], Y):
        X = [torch.as_tensor(x) for x in X]
        self.train_X.extend(X)
        self.train_Y.extend(Y)
        for x, y in zip(X, Y):
            if isinstance(x, Tensor):
                x = x.cpu().detach().numpy()
            self.fillin_strategy.update(x.reshape(1, -1), y)
