import numpy as np
from typing import Set
import logging

log = logging.getLogger(__name__)


def cal_idx(i, j, dims):
    i, j = min(i, j), max(i, j)
    return int( (2*dims-i-1)*i/2 + j-i-1 )


class OrderConstraint:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def is_feasible(self, x, feature_x):
        if x[self.first] < x[self.second]:
            return True
        else:
            return False
        
    def __str__(self):
        return 'order constraint: {} is before {}'.format(self.first, self.second)


class PositionConstraint:
    def __init__(self, idx, feasible_pos: Set):
        self.idx = idx
        self.feasible_pos = feasible_pos

    def is_feasible(self, x, feature_x):
        if x[self.idx] in self.feasible_pos:
            return True
        else:
            return False
        
    def __str__(self):
        return 'position constaint: {} should be {}'.format(self.idx, self.feasible_pos)


class Node:
    def __init__(self, dims, parent, leaf_size, max_depth):
        self.dims = dims
        self.parent = parent
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.depth = 0 if parent is None else parent.depth + 1
        # v and n for UCB
        self.v = None
        self.n = 0

        self.constraints = {'order': [], 'position': []}
        
        self.kids = [] # 0: good, 1: bad

        self.train_X = []
        self.feature_X = []
        self.train_Y = []

    def get_uct(self, Cp):
        if self.parent == None:
            return float('inf')
        if self.n == 0:
            return float('inf')
        return self.v + 2*Cp*np.sqrt( 2* np.power(self.parent.n, 0.5) / self.n )

    def is_splittable(self):
        if len(self.kids) > 0:
            return False
        if len(self.train_X) < self.leaf_size:
            return False
        if self.depth is not None and self.depth >= self.max_depth:
            return False
        return True

    def update_bag(self, train_X, feature_X, train_Y):
        self.train_X.extend(train_X)
        self.feature_X.extend(feature_X)
        self.train_Y.extend(train_Y)

        self.v = np.mean(self.train_Y)

    def add_constraint(self, constraint_type, constraint):
        self.constraints[constraint_type].append(constraint)

    def divide_data(self, train_X, feature_X, train_Y):
        mean_y = np.mean(train_Y)

        good_idx = [idx for idx, y in enumerate(train_Y) if y >= mean_y]
        good_train_X, good_feature_X, good_train_Y = [], [], []
        for idx in good_idx:
            good_train_X.append(train_X[idx])
            good_feature_X.append(feature_X[idx])
            good_train_Y.append(train_Y[idx])

        bad_idx = [idx for idx, y in enumerate(train_Y) if y < mean_y]
        bad_train_X, bad_feature_X, bad_train_Y = [], [], []
        for idx in bad_idx:
            bad_train_X.append(train_X[idx])
            bad_feature_X.append(feature_X[idx])
            bad_train_Y.append(train_Y[idx])
        
        assert len(good_train_X) + len(bad_train_X) == len(train_X)
        assert len(good_train_X) != 0 and len(good_train_X) != len(train_X)
        
        return good_train_X, good_feature_X, good_train_Y, \
            bad_train_X, bad_feature_X, bad_train_Y

    def find_common_pattern(self, pattern_type, train_X, feature_X, 
        good_train_X, good_feature_X, bad_train_X, bad_feature_X, n=1):
        # 现在只能找到good里面的特征，但可能bad也满足，所以
        # 感觉还是需要找good和bad存在的差别，作为constraint
        # 先找到差别，然后看good中的顺序是什么样的
        # 这里可以考虑同时加多个限制，只保留能区分的限制

        # find the common pattern in the data, and use the pattern as a 
        # constraint of the good data
        assert pattern_type in ['order', 'position']
        if pattern_type == 'order':
            # score = np.abs(np.sum(feature_X, axis=0))
            # max_idx = np.argmin(score)

            good_score = np.sum(good_feature_X, axis=0)
            bad_score = np.sum(bad_feature_X, axis=0)
            diff = np.abs(good_score - bad_score)
            max_idx = np.argmax(diff)
            
            for i in range(self.dims):
                cnt = (2*self.dims-i-1)*i/2
                if cnt > max_idx:
                    break
            i = i-1
            cnt = (2*self.dims-i-1)*i/2
            j = int(max_idx - cnt + i + 1)

            # log.info('pattern: {}, {}, {}'.format(max_idx, i, j))
            assert cal_idx(i, j, self.dims) == max_idx

            constraint = OrderConstraint(i, j)

            # check whether the constraint can idensify good and bad data
            total = len(good_train_X)
            feasible = 0
            for x in good_train_X:
                feasible += constraint.is_feasible(x, None)
            if feasible / total < 0.5:
                constraint = OrderConstraint(j, i)
        elif pattern_type == 'position':
            good_X_dict = np.zeros((self.dims, self.dims))
            for i in range(len(good_train_X)):
                for j in range(self.dims):
                    v = good_train_X[i][j]
                    good_X_dict[j][v] += 1
            bad_X_dict = np.zeros((self.dims, self.dims))
            for i in range(len(bad_train_X)):
                for j in range(self.dims):
                    v = bad_train_X[i][j]
                    bad_X_dict[j][v] += 1
            score = [0 for i in range(self.dims)]
            for i in range(self.dims):
                score[i] = np.sum((good_X_dict[i] - bad_X_dict[i])**2)
            idx = np.argmax(score)
            feasible_pos = set(good_train_X[idx])

            constraint = PositionConstraint(idx, feasible_pos)
        else:
            raise NotImplementedError
        return constraint

    def is_constraint_useful(self, constraint, train_X, feature_X, train_Y):
        good_train_X, good_feature_X, good_train_Y = [], [], []
        bad_train_X, bad_feature_X, bad_train_Y = [], [], []
        for x, feature_x, y in zip(train_X, feature_X, train_Y):
            is_feasible = constraint.is_feasible(x, feature_x)
            if is_feasible:
                good_train_X.append(x)
                good_feature_X.append(feature_x)
                good_train_Y.append(y)
            else:
                bad_train_X.append(x)
                bad_feature_X.append(feature_x)
                bad_train_Y.append(y)
        
        if len(good_train_X) == 0 or len(good_train_X) == len(train_X):
            return False
        elif np.mean(good_train_Y) <= np.mean(bad_train_Y):
            return False
        else:
            return True

    def classify(self, constraint, train_X, feature_X, train_Y):
        good_train_X, good_feature_X, good_train_Y = [], [], []
        bad_train_X, bad_feature_X, bad_train_Y = [], [], []
        for x, feature_x, y in zip(train_X, feature_X, train_Y):
            is_feasible = constraint.is_feasible(x, feature_x)
            if is_feasible:
                good_train_X.append(x)
                good_feature_X.append(feature_x)
                good_train_Y.append(y)
            else:
                bad_train_X.append(x)
                bad_feature_X.append(feature_x)
                bad_train_Y.append(y)
        return good_train_X, good_feature_X, good_train_Y, bad_train_X, bad_feature_X, bad_train_Y

    def split(self):
        log.info('split')
        good_train_X, good_feature_X, good_train_Y, bad_train_X, bad_feature_X, bad_train_Y \
            = self.divide_data(self.train_X, self.feature_X, self.train_Y)
        pattern_type = 'order'
        # pattern_type = 'position'
        # constraint = self.find_common_pattern(pattern_type, good_train_X, good_feature_X)
        constraint = self.find_common_pattern(pattern_type, self.train_X, self.feature_X, 
            good_train_X, good_feature_X, bad_train_X, bad_feature_X)
        if pattern_type == 'order':
            inv_constraint = OrderConstraint(constraint.second, constraint.first)
        elif pattern_type == 'position':
            inv_constraint = PositionConstraint(constraint.idx, set(range(self.dims)) - constraint.feasible_pos)
        else:
            raise NotImplementedError

        if not self.is_constraint_useful(constraint, self.train_X, self.feature_X, self.train_Y):
            constraint, inv_constraint = inv_constraint, constraint
            log.info('not useful')

        good_train_X, good_feature_X, good_train_Y, bad_train_X, bad_feature_X, bad_train_Y = \
            self.classify(constraint, self.train_X, self.feature_X, self.train_Y)

        assert len(good_train_X) != 0 and len(good_train_X) != len(self.train_X)
        assert np.mean(good_train_Y) > np.mean(bad_train_Y)
        
        good_kid = Node(self.dims, self, self.leaf_size)
        bad_kid = Node(self.dims, self, self.leaf_size)

        # add constraints to the new node
        for cons_type in self.constraints.keys():
            for cons in self.constraints[cons_type]:
                good_kid.add_constraint(cons_type, cons)
                bad_kid.add_constraint(cons_type, cons)
        good_kid.add_constraint(pattern_type, constraint)
        log.info('add good constraints: {}'.format(constraint))
        bad_kid.add_constraint(pattern_type, inv_constraint)
        
        good_kid.update_bag(good_train_X, good_feature_X, good_train_Y)
        bad_kid.update_bag(bad_train_X, bad_feature_X, bad_train_Y)
        self.kids = [good_kid, bad_kid]
        
        return good_kid, bad_kid
