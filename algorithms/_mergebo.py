# Paper: FROM SORTING ALGORITHMS TO SCALABLE KERNELS: BAYESIAN OPTIMIZATION IN HIGHDIMENSIONAL PERMUTATION SPACES
# ICLR'26 submission: https://openreview.net/forum?id=7QtKdabBP9

import torch
import gpytorch
import numpy as np
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from gpytorch.kernels import Kernel
import logging
from copy import deepcopy
from typing import List
from collections import deque
from ._base import BaseOptimizer
from ._fillin_strategy import PermutationRandomStrategy, PermutationBestKPosStrategy
from ._utils import get_init_samples, select, get_subset
from ._ea import EA
import time

# 设置日志
log = logging.getLogger(__name__)

class MergeKernel(Kernel):
    """Merge kernel for permutations - 支持3D输入"""
    has_lengthscale = True
    
    def forward(self, X, X2=None, diag=False, **params):
        """
        Args:
            X: shape (batch_shape1, n1, d) or (n1, d)
            X2: shape (batch_shape2, n2, d) or (n2, d) or None
            diag: 是否只计算对角线
        """
        if X2 is None:
            X2 = X
        
        # 确保都是3D张量以便广播
        if X.dim() == 2:
            X = X.unsqueeze(0)
        if X2.dim() == 2:
            X2 = X2.unsqueeze(0)
        
        # 计算成对距离
        # X: (b1, n1, d), X2: (b2, n2, d)
        # 使用广播计算所有对之间的差异
        diff = X.unsqueeze(-2) - X2.unsqueeze(-3)  # (b1, n1, 1, d) - (b2, 1, n2, d) -> (b1, b2, n1, n2, d)
        
        # 平方和
        squared_dist = torch.sum(diff ** 2, dim=-1)  # (b1, b2, n1, n2)
        
        # 应用长度尺度
        result = torch.exp(-self.lengthscale * squared_dist)
        
        # 处理对角线情况
        if diag:
            if X.shape[:-2] != X2.shape[:-2] or X.shape[-2] != X2.shape[-2]:
                raise RuntimeError("For diag=True, X and X2 must have same shape")
            return torch.diagonal(result, dim1=-2, dim2=-1)  # (b, n)
        
        # 如果只有一个批次维度，移除多余的维度
        if X.shape[0] == 1 and X2.shape[0] == 1:
            result = result.squeeze(0).squeeze(0)  # (n1, n2)
        elif X.shape[0] == 1:
            result = result.squeeze(0)  # (b2, n1, n2)
        elif X2.shape[0] == 1:
            result = result.squeeze(1)  # (b1, n1, n2)
        
        return result

# def merge_sort(arr):
#     # print(f'Handling array: {arr}')
#     if len(arr) == 1:
#         return []
#     if len(arr) > 1:
#         mid = len(arr) // 2
#         left_half = deepcopy(arr[:mid])
#         right_half = deepcopy(arr[mid:])

#         left_feature = merge_sort(left_half)
#         right_feature = merge_sort(right_half)

#         # print(f'L & R: {left_feature}, {right_feature}')
#         feature = [] + left_feature + right_feature

#         i = j = k = 0

#         # Merge the two halves into the original list
#         while i < len(left_half) and j < len(right_half):
#             if left_half[i] < right_half[j]:
#                 arr[k] = left_half[i]
#                 i += 1
#                 feature.append(-1)
#             else:
#                 arr[k] = right_half[j]
#                 j += 1
#                 feature.append(1)
#             k += 1

#         while i < len(left_half):
#             arr[k] = left_half[i]
#             i += 1
#             k += 1
#             feature.append(1)

#         while j < len(right_half):
#             arr[k] = right_half[j]
#             j += 1
#             k += 1
#             feature.append(-1)

#         # print('Return: ', feature)
#         return feature[:-1]

# def featurize_merge(x):
#     """
#     Featurize the permutation vector into continuous space using merge kernel. Only available to permutation vector.
#     """
#     if x.dim() == 1:
#         x = x.unsqueeze(0)
#     assert len(x.shape) == 2, "Only featurize 2 dimension permutation vector"
#     feature = []
#     x_copy = deepcopy(x)
#     for arr in x_copy:
#         feature.append(merge_sort(arr))
#     normalizer = np.sqrt(x.size(1)*(x.size(1) - 1)/2)
#     return torch.tensor(feature)/normalizer

def merge_sort(arr, left=0, right=None, feature=None):
    """
    优化版的归并排序特征提取
    使用原地操作和索引，避免深度拷贝
    """
    if feature is None:
        feature = []
    if right is None:
        right = len(arr)
    
    if right - left <= 1:
        return feature
    
    # 计算中点
    mid = (left + right) // 2
    
    # 递归处理左右两部分
    merge_sort(arr, left, mid, feature)
    merge_sort(arr, mid, right, feature)
    
    # 创建临时数组用于合并
    temp = arr[left:right]
    i = left
    j = mid
    k = 0
    
    # 合并两个有序部分
    while i < mid and j < right:
        if arr[i] < arr[j]:
            temp[k] = arr[i]
            i += 1
            feature.append(-1)
        else:
            temp[k] = arr[j]
            j += 1
            feature.append(1)
        k += 1
    
    # 处理剩余元素
    while i < mid:
        temp[k] = arr[i]
        i += 1
        k += 1
        feature.append(1)
    
    while j < right:
        temp[k] = arr[j]
        j += 1
        k += 1
        feature.append(-1)
    
    # 将临时数组复制回原数组
    arr[left:right] = temp
    
    # 移除最后一个多余的feature
    if len(feature) > 0:
        feature.pop()
    
    return feature

def featurize_merge(x):
    """
    优化后的特征化函数
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    assert len(x.shape) == 2, "Only featurize 2 dimension permutation vector"
    
    feature_list = []
    batch_size, n = x.shape
    
    # 预计算归一化因子
    normalizer = (n * (n - 1) / 2) ** 0.5
    
    # 处理每个样本
    for i in range(batch_size):
        # 转换为Python列表（避免在递归中操作torch tensor）
        arr = x[i].tolist()
        # 提取特征
        feat = merge_sort(arr, 0, len(arr), [])
        feature_list.append(feat)
    
    # 找到最大长度并填充（如果需要统一尺寸）
    max_len = max(len(f) for f in feature_list)
    if max_len > 0:
        padded_features = []
        for feat in feature_list:
            if len(feat) < max_len:
                # 使用0填充
                padded = feat + [0] * (max_len - len(feat))
                padded_features.append(padded)
            else:
                padded_features.append(feat)
        feature_tensor = torch.tensor(padded_features, dtype=torch.float32)
    else:
        feature_tensor = torch.tensor(feature_list, dtype=torch.float32)
    
    return feature_tensor / normalizer


class FeatureCacheMerge:
    """Cache for featurized permutations"""
    def __init__(self, max_size=500):
        self.cache = {}
        self.max_size = max_size
        self.key_queue = deque()
    
    def _get_key(self, x):
        """Convert tensor to hashable key"""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return tuple(x.flatten())
    
    def get(self, x):
        """Get cached feature or None"""
        key = self._get_key(x)
        return self.cache.get(key, None)
    
    def push(self, x):
        key = self._get_key(x)
        if key not in self.cache:
            feature = featurize_merge(x)
            self.cache[key] = feature
            self.key_queue.append(key)
            
            # 使用LRU策略
            if len(self.cache) > self.max_size:
                old_key = self.key_queue.popleft()
                if old_key in self.cache:
                    del self.cache[old_key]
        return self.cache[key]

class MergeBO(BaseOptimizer):
    """Bayesian Optimization over Permutations using Merge kernel"""
    
    def __init__(
        self, 
        dims, 
        lb, 
        ub, 
        n_init=20,
        init_sampler_type='permutation',
        kernel_type='merge',
        acqf_opt_n_restarts=10,
        ls_steps=100,
        active_dims=None,
        fillin_type='random',
        device='cpu'
    ):
        """
        Args:
            dims: problem dimension (length of permutation)
            lb: lower bounds (not used for permutations)
            ub: upper bounds (not used for permutations)
            n_init: number of initial samples
            kernel_type: type of kernel ('merge' only)
            acqf_opt_n_restarts: number of random restarts for acquisition optimization
            ls_steps: number of local search steps
            active_dims: number of active dimensions (None for all)
            fillin_type: type of fill-in strategy ('random' or 'best_pos')
            device: device to run on ('cpu' or 'cuda')
        """
        self.dims = dims
        self.lb = np.ones(self.dims) * lb
        self.ub = np.ones(self.dims) * ub
        self.n_init = n_init
        self.init_sampler_type = init_sampler_type
        self.kernel_type = kernel_type
        self.acqf_opt_n_restarts = acqf_opt_n_restarts
        self.ls_steps = ls_steps
        self.active_dims = None if active_dims is None else min(active_dims, self.dims) # If None, use all dimensions
        self.fillin_type = fillin_type
        
        # fill in strategy
        fillin_strategy_factory = {
            'random': PermutationRandomStrategy(dims, lb, ub),
            'best_pos': PermutationBestKPosStrategy(dims, lb, ub, 10),
        }
        if self.active_dims is not None:
            self.fillin_strategy = fillin_strategy_factory[fillin_type]

        # 设置设备
        self.device = torch.device(device)
        if device == 'cuda' and not torch.cuda.is_available():
            log.warning("CUDA not available, using CPU")
            self.device = torch.device('cpu')
        
        # 状态变量
        self.train_x = []  # List of tensors (permutations)
        self.train_y = []  # List of objective values
        self.cache = FeatureCacheMerge()
        
        # 缓存待评估的点
        self.cache_X = deque()
    
    def get_ckpt_dict(self):
        return {
            'train_x': self.train_x,
            'train_y': self.train_y,
        }
    
    def load_ckpt_dict(self, ckpt_dict):
        self.train_x = ckpt_dict['train_x']
        self.train_y = ckpt_dict['train_y']
        best_idx = np.argmax(self.train_y)
        best_x = self.train_x[best_idx]
        best_y = self.train_y[best_idx]
        return best_x, best_y

    def _init_samples(self, init_sampler_type, n) -> List[np.ndarray]:
        points = get_init_samples(init_sampler_type, n, self.dims, self.lb, self.ub)
        points = [points[i] for i in range(len(points))]
        return points
    
    def _initialize_model(self, train_x_feat, train_y):
        """Initialize and train GP model"""
        # Normalize objective values
        train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
        train_y_normalized = (train_y_tensor - train_y_tensor.mean()) / (train_y_tensor.std() + 1e-8)
        train_y_normalized = train_y_normalized.unsqueeze(-1).to(self.device)
        
        # Create kernel
        if self.kernel_type == 'merge':
            covar_module = MergeKernel().to(self.device)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

        # Create GP model
        model = SingleTaskGP(train_x_feat, train_y_normalized, covar_module=covar_module).to(self.device)
        
        # Create likelihood with fixed noise
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        
        # Create marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)

        # Set fixed noise for model and likelihood
        model.likelihood.noise_covar.noise = torch.tensor(0.0001).float()
        mll.model.likelihood.noise_covar.raw_noise.requires_grad = False
        # Train model
        fit_gpytorch_model(mll)
        
        return model, mll
    
    def _optimize_acqf_ea(self, AF):
        """Optimize acquisition function using EA"""
        if self.active_dims is None:
            ea_alg = EA(self.dims, self.lb, self.ub, pop_size=10, init_sampler_type='permutation', mutation_type='insert', crossover_type='pmx')
        else:
            ea_alg = EA(self.active_dims, self.lb, self.ub, pop_size=10, init_sampler_type='permutation', mutation_type='insert', crossover_type='pmx')

        for _ in range(500):
            cands = ea_alg.ask()

            # Convert to tensor
            cands_tensor = torch.tensor(np.array(cands), dtype=torch.float32, device=self.device)
            # Featurize and compute acquisition values
            cands_feat = featurize_merge(cands_tensor).to(self.device).unsqueeze(1)
            with torch.no_grad():
                acq_vals = AF(cands_feat)
            cands_y = acq_vals.cpu().numpy().tolist()

            ea_alg.tell(cands, cands_y)

            del cands_tensor, cands_feat, acq_vals
        
        # 返回最优解
        indices = np.argsort(ea_alg.fitness)[-1]
        proposed_X = torch.from_numpy(ea_alg.population[indices])
        proposed_Y = ea_alg.fitness[indices]
        return proposed_X, proposed_Y

    def _optimize_acquisition(self, model, train_y):
        """Optimize acquisition function"""
        # Create acquisition function
        best_f = torch.tensor(train_y).max().item()
        EI = ExpectedImprovement(model, best_f=best_f).to(self.device)
        candidate, _ = self._optimize_acqf_ea(EI)
        return candidate
    
    def ask(self, n_repeats=1):
        """Get next point to evaluate"""
        if len(self.cache_X) + len(self.train_x) < self.n_init:
            points = self._init_samples(self.init_sampler_type, self.n_init)
            self.cache_X.extend(points)
        
        if len(self.cache_X) > 0:
            return [self.cache_X.popleft()]
        
        if self.active_dims is not None:
            # 使用 Dropout 的方式，随机选择 active_dims 个维度，取对应的特征作为 subset_X
            idx = select(self.dims, self.active_dims)
            train_X_tensor = torch.vstack(self.train_x).float().to(self.device)
            subset_X = get_subset(train_X_tensor, idx).to(self.device)
            # Prepare training data
            train_x_feat = []
            for x in subset_X:
                feat = self.cache.get(x)
                if feat is None:
                    feat = self.cache.push(x)
                train_x_feat.append(feat)
        else:
            # Prepare training data
            train_x_feat = []
            for x in self.train_x:
                feat = self.cache.get(x)
                if feat is None:
                    feat = self.cache.push(x)
                train_x_feat.append(feat)
        
        train_x_feat = torch.cat(train_x_feat, dim=0).to(self.device)
        train_y = self.train_y
        
        # Train model
        model, mll = self._initialize_model(train_x_feat, train_y)
        
        # Optimize acquisition function
        next_point = self._optimize_acquisition(model, train_y)
        
        if self.active_dims is not None:
            # fill in
            fixed_vars = {j: pos for j, pos in zip(idx, next_point)}
            next_point = self.fillin_strategy.fillin(fixed_vars)
            next_point = torch.from_numpy(next_point)

        # Check if point already evaluated
        if any(torch.all(torch.eq(next_point, x)) for x in self.train_x):
            log.warning("Proposed point already evaluated, generating random permutation")
            next_point = torch.from_numpy(np.random.permutation(self.dims))
        
        return [next_point.cpu().numpy()]
    
    def tell(self, X, Y):
        """Provide evaluation results"""
        for x, y in zip(X, Y):
            if self.active_dims is not None:
                self.fillin_strategy.update(x.reshape(1, -1), y)
            # Convert to tensor
            x_tensor = torch.from_numpy(x).to(torch.int64)
            
            # Store in training data
            self.train_x.append(x_tensor)
            self.train_y.append(float(y))
            
    