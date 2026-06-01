import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from typing import Set
import logging
from ._utils import featurize

log = logging.getLogger(__name__)


def permutation_distance(x1, x2, **kwargs):
    f_x1 = featurize(x1, 'numpy')
    f_x2 = featurize(x2, 'numpy')
    return np.sum((f_x1 - f_x2)**2)


class Node:
    def __init__(self, dims, parent, leaf_size, max_depth):
        self.dims = dims
        self.parent = parent
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.depth = 0 if parent is None else parent.depth + 1
        self.classify = KNeighborsClassifier(n_neighbors=3, metric=permutation_distance, metric_params={}, n_jobs=10)
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

    def clustering(self, train_X, feature_X, train_Y):
        # 0: good cluster, 1: bad cluster
        mean_y = np.mean(train_Y)
        good_idx = [idx for idx, y in enumerate(train_Y) if y >= mean_y]
        bad_idx = [idx for idx, y in enumerate(train_Y) if y < mean_y]
        label = []
        for i in range(len(train_X)):
            if i in good_idx:
                label.append(0)
            else:
                label.append(1)
        return label

    def split(self):
        # log.info('split')
        # good_train_X, good_feature_X, good_train_Y, bad_train_X, bad_feature_X, bad_train_Y \
        #     = self.divide_data(self.train_X, self.feature_X, self.train_Y)
        label = self.clustering(self.train_X, self.feature_X, self.train_Y)
        
        # train classify
        print('train X')
        print(self.train_X)
        print('label')
        print(label)
        self.classify.fit(self.train_X, label)
        
        pred = self.classify.predict(self.train_X)
        print('pred')
        print(pred)
        good_train_X, good_feature_X, good_train_Y, bad_train_X, bad_feature_X, bad_train_Y = \
            [], [], [], [], [], []
        for i in range(len(self.train_X)):
            if pred[i] == 0:
                good_train_X.append(self.train_X[i])
                good_train_Y.append(self.train_Y[i])
            elif pred[i] == 1:
                bad_train_X.append(self.train_X[i])
                bad_train_Y.append(self.train_Y[i])

        # assert len(good_train_X) != 0 and len(good_train_X) != len(self.train_X)
        # assert np.mean(good_train_Y) > np.mean(bad_train_Y)

        if len(good_train_X) == 0 or len(good_train_X) == len(self.train_X):
            return None, None
        else:
            good_kid = Node(self.dims, self, self.leaf_size, self.leaf_size)
            bad_kid = Node(self.dims, self, self.leaf_size, self.leaf_size)

            good_kid.update_bag(good_train_X, good_feature_X, good_train_Y)
            bad_kid.update_bag(bad_train_X, bad_feature_X, bad_train_Y)
            self.kids = [good_kid, bad_kid]
            
            return good_kid, bad_kid
