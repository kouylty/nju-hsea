import numpy as np
from typing import List
import time
import logging
from collections import deque
from ._base import BaseOptimizer
# from ._node import Node
from ._node_knn import Node
from ._ea_operator import swap_mutation
from ._utils import get_init_samples, featurize, FeatureCache

log = logging.getLogger(__name__)

# 记录找到差的解的次数，从而重建树
# 控制树的深度
# 现在运行过程中完全不会划分，只在重构树的时候才会划分，也就是每200次评估


class MCTS(BaseOptimizer):
    def __init__(self, dims, lb, ub, Cp=1, leaf_size=20, max_depth=20, max_propose=200, solver_type='sa'):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.Cp = Cp
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.max_propose = max_propose

        self.solver_type = solver_type
        self.solver_config = {
            'T': 100,
            'decay': 0.99,
            'update_freq': 100,
        }
        self.reset_solver()

        self.train_X = []
        self.feature_X = []
        self.feature_cache = FeatureCache('numpy')
        self.train_Y = []
        self.best_x = None
        self.best_y = None
        self.n_propose = 0

        self.nodes = []
        root = Node(self.dims, None, self.leaf_size, self.max_depth)
        self.nodes.append(root)
        self.root = root

    def populate_training_data(self):
        self.nodes.clear()
        root = Node(self.dims, None, self.leaf_size, self.max_depth)
        self.nodes.append(root)
        self.root = root
        self.root.update_bag(self.train_X, self.feature_X, self.train_Y)

    # def get_split_idx(self):
    #     split_idx = []
    #     for idx, node in enumerate(self.nodes):
    #         if node.is_splittable():
    #             split_idx.append(idx)
    #     return split_idx

    # def is_splittable(self):
    #     split_idx = self.get_split_idx()
    #     if len(split_idx) > 0:
    #         return True
    #     else:
    #         return False

    def dynamic_treeify(self):
        self.populate_training_data()
        
        to_split_idx = deque()
        if self.nodes[0].is_splittable():
            to_split_idx.append(0)
        while len(to_split_idx) > 0:
            idx = to_split_idx.popleft()
            parent = self.nodes[idx]
            good_kid, bad_kid = parent.split()
            if good_kid is None and bad_kid is None:
                pass
            else:
                self.nodes.append(good_kid)
                if good_kid.is_splittable():
                    to_split_idx.append(len(self.nodes)-1)
                self.nodes.append(bad_kid)
                if bad_kid.is_splittable():
                    to_split_idx.append(len(self.nodes)-1)

    def select(self):
        curr_node = self.root
        path = []
        while len(curr_node.kids) > 0:
            uct = [node.get_uct(self.Cp) for node in curr_node.kids]
            choice = np.random.choice(np.argwhere(uct == np.amax(uct)).reshape(-1), 1)[0]
            path.append( (curr_node, choice) )
            curr_node = curr_node.kids[choice]
        return curr_node, path

    def is_feasible(self, x, leaf, path):
        for node, label in path:
            pred = node.classify.predict([x])
            if pred != label:
                return False
        return True

    def optimize(self, node, path):
        if len(self.train_X) == 0:
            next_x = get_init_samples('permutation', 1, self.dims, self.lb, self.ub)[0]
            next_feature_x = self.feature_cache.push(next_x)
            return next_x, next_feature_x

        # obtain the best in the node
        best_idx = np.argmax(node.train_Y)
        best_x = node.train_X[best_idx]

        if self.solver_type == 'sa':
            while True:
                next_x = swap_mutation(best_x)
                next_feature_x = self.feature_cache.push(next_x)
                if self.is_feasible(next_x, node, path):
                    break
            # 没有做sa中的按概率接受差的解
            self.solver_state['step'] += 1
            if self.solver_state['step'] % self.solver_config['update_freq'] == 0:
                self.solver_state['T'] *= self.solver_config['decay']
        elif self.solver_type == 'random':
            raise NotImplementedError
        else:
            raise NotImplementedError

        return next_x, self.feature_cache.push(next_x)

    def backpropogate(self, leaf, X, feature_X, Y):
        curr_node = leaf
        while curr_node is not None:
            curr_node.v = (curr_node.v * curr_node.n + np.sum(Y)) / (curr_node.n + len(Y))
            curr_node.n += len(Y)
            curr_node.update_bag(X, feature_X, Y)
            curr_node = curr_node.parent

    def ask(self, n_repeats=1) -> List[np.ndarray]:
        if self.n_propose == 0:
            self.dynamic_treeify()
            self.leaf, self.path = self.select()
            self.reset_solver()
        next_x, next_feature_x = self.optimize(self.leaf, self.path)
        # self.print_tree()
        return [next_x]

    def tell(self, X: List[np.ndarray], Y):
        self.n_propose = (self.n_propose + 1) % self.max_propose
        self.train_X.extend(X)
        feature_X = [self.feature_cache.push(x) for x in X]
        self.feature_X.extend(feature_X)
        self.train_Y.extend(Y)

        self.backpropogate(self.leaf, X, feature_X, Y)

    def reset_solver(self):
        self.solver_state = {
            'T': self.solver_config['T'],
            'step': 0,
        }
        
    def print_tree(self):
        for idx, node in enumerate(self.nodes):
            log.info('idx: {}'.format(idx))
            if node.parent is not None:
                log.info('is good node: {}'.format(node.parent.kids[0] is node))