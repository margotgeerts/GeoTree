import numpy as np
import matplotlib.pyplot as plt
from ._splits import *
from time import time
import joblib
import warnings

TREE_BALANCE_BIAS = 0
DEBUG_SPLITS = True
EPSILON = np.finfo('double').eps


def mse(y_true, y_pred):
    return np.sum((y_pred - y_true)**2) / len(y_true)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def mape(y_true,y_pred):
    return np.mean(np.abs((y_pred-y_true))/np.maximum(np.abs(y_true),EPSILON))

def r2_score(y_true,y_pred):
    y_bar = np.sum(y_true) / len(y_true)
    ssr = np.sum((y_pred-y_true)**2)
    sst = np.sum((y_bar-y_true)**2)
    return 1 - (ssr/sst)

def calc_metrics(y_true, y_pred, string=""):
    metrics = {}
    rmsey = rmse(y_true, y_pred)
    maey = mae(y_true, y_pred)
    mapey = mape(y_true,y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics['rmse'+string] = rmsey
    metrics['mae'+string] = maey
    metrics['mape'+string] = mapey
    metrics['r2'+string] = r2
    return metrics

def balance_correction(k, num):
    x = np.linspace(0, 1, num+2)
    c = .00001; x[0] += c; x[-1] -= c
    r = .5 + np.log(x / (1 - x)) / k
    return np.clip(r[1:-1], 0, 1)

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def evaluate_split(candidate_split, X, y, parent_mse, n):
    if candidate_split is None:
        return -1
    tidx = candidate_split.is_true(X)
    left_y, right_y = y[~tidx], y[tidx]
    if not len(left_y) or not len(right_y):
        return 0 # Zou niet mogen voorvallen
    left_mse = mse(left_y, np.mean(left_y))
    right_mse = mse(right_y, np.mean(right_y))
    n_t = len(y)
    left_ratio = len(left_y) / n_t
    right_ratio = len(right_y) / n_t
    if TREE_BALANCE_BIAS > 0:
        err = balance_correction(TREE_BALANCE_BIAS, n_t)
        left_ratio = err[len(left_y)]
        right_ratio = 1 - left_ratio
    weight = n_t / n
    gain = weight * (parent_mse - (left_mse * left_ratio) - (right_mse * right_ratio))
    return np.float64(gain)


def evaluate_splits(generator, X, y, parent_mse, features, geo_features, n, bbox, random_state):
    best_mse_gain, best_split = 0.0, None
    #if geo_features is None:
    #    geo_features = X.shape[1]
    gen = generator.generate_candidates(X=X, y=y, parent_mse=parent_mse, features=features, geo_features=geo_features, n=n, bbox=bbox, random_state=random_state)
    if gen is None:
        return None, None
    for candidate_split in gen:
        if candidate_split is None:
            continue
        elif hasattr(candidate_split,'feat'):
            if candidate_split.feat is None:
                continue
        elif ((candidate_split.feat1 is None) or (candidate_split.feat2 is None)):
            continue
        gain = evaluate_split(candidate_split, X, y, parent_mse, n)
        if gain == 0.0:
            print(f'zero gain: {gain}, {candidate_split}')
        if gain >= best_mse_gain:
            best_mse_gain = gain
            best_split = candidate_split
    #if (best_split is None) and DEBUG_SPLITS:
    #    print('no split found', type(generator), len(X))
    return best_mse_gain, best_split


class Node:
    def __init__(self, tree, idx, depth, parent=None):
        self.tree = tree
        self.idx = idx
        self.depth = depth
        self.parent = parent
        self.left = None
        self.right = None
        self.split = None
        self.gain = None
        self.continue_growing = True
        self.count_none = 0
        self.min_gain = 0.0
        

    @property
    def X(self):
        return self.tree.X[self.idx]
    
    @property
    def y(self):
        return self.tree.y[self.idx]

    @property
    def yhat(self):
        return np.mean(self.tree.y[self.idx]) if (isinstance(self.tree.y,(list,np.ndarray)) and  len(self.tree.y)>1) else self.tree.y
    
    @property
    def mse(self):
        return mse(self.y, self.yhat)

    def is_not_split(self):
        return self.left is None and self.right is None

    def is_leaf(self):
        return ((self.tree.max_depth and self.depth >= self.tree.max_depth) or # max depth reached
                  len(self.idx) < 2 or # only one sample in node
                  ((self.gain is not None) and self.gain <= self.min_gain) or # tried to find split but gain is 0
                  self.mse <= EPSILON or    # mse is 0
                  np.all(self.X==self.X[0])) # all features are constant
    
    def best_generator_split(self, generator, features, geo_features, n_RESTART, geosplits, min_area, random_state):
        if (geosplits is not None) and isinstance(generator, geosplits) and (PolyArea(self.X[:,0],self.X[:,1]) > min_area):
            return 0, None
        gain, split = evaluate_splits(generator, self.X, self.y, self.mse, features, geo_features, len(self.tree.y), self.tree.bbox, random_state)
        if (split is None) and (n_RESTART>0):
            count=0
            while (split is None) and (count < n_RESTART):
                gain, split = evaluate_splits(generator, self.X, self.y, self.mse, features, geo_features, len(self.tree.y), self.tree.bbox, random_state)
                count += 1
        return gain, split
    
    def best_split(self, generators, features, geo_features, n_RESTART, geosplits, min_area, n_jobs, random_state):
        assert generators

        best_mse_gain, best_split = 0, None
        if np.all(self.X[:,features]==self.X[0,features]):
            return best_mse_gain, best_split

        gains, splits = zip(*joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self.best_generator_split)(generator,
                                                                                                 features,
                                                                                                 geo_features,
                                                                                                 n_RESTART,
                                                                                                 geosplits,
                                                                                                 min_area,
                                                                                                 random_state) for generator in generators))
        
        best_mse_gain = np.max(gains)
        best_split = splits[np.argmax(gains)]

        if DEBUG_SPLITS:
            print(TREE_BALANCE_BIAS, best_mse_gain, best_split, len(self.tree.get_leafs()))

        return best_mse_gain, best_split
    
    def get_left_right_idx(self, X=None, idx=None):
        if X is None:
            X = self.X
            idx = self.idx
        if self.split is None:
            return None, None
        tidx = self.split.is_true(X)
        idx_left = idx[np.where(~tidx)[0]]
        idx_right = idx[np.where(tidx)[0]]
        return idx_left, idx_right

    def grow(self, generators, features, geo_features, CONTINUE_NONE=True, n_RESTART=0, geosplits=None, min_area=0.0, min_gain=0.0, n_jobs=None, random_state=None):
        self.min_gain = min_gain
        if self.continue_growing:
            best_mse_gain, best_split = self.best_split(generators, features, geo_features, n_RESTART, geosplits, min_area, n_jobs, random_state)
            if best_split is None:
                self.count_none += 1
                self.continue_growing = CONTINUE_NONE
            #if best_mse_gain*(len(self.y)/len(self.tree.y)) >= min_gain:
            self.split = best_split
            self.gain = best_mse_gain
            idx_left, idx_right = self.get_left_right_idx()
            if idx_left is not None:
                self.left = Node(self.tree, idx_left, self.depth+1, self)
            if idx_right is not None:
                self.right = Node(self.tree, idx_right, self.depth+1, self)
        elif DEBUG_SPLITS:
            print('stop growing node')

class Tree:
    def __init__(self, X, y, max_depth):
        self.root = Node(self, np.arange(0, len(X)), 0)
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.X_test = None
        self.y_test = None
        self.metrics = None
        self.curr_depth = 0
        
    @property
    def bbox(self):
        if self.X.shape[1] > 1:
            return (max(self.X[:,0]) - min(self.X[:,0])) * (max(self.X[:,1]) - min(self.X[:,1]))
        else:
            return 0.0

    @property
    def n_leaves(self):
        return len(self.get_leafs())

    @property
    def node_count(self):
        return len(self.get_nodes())

    def get_nodes(self):
        nodes = [self.root]
        nodes_parsed = []
        while nodes:
            node = nodes.pop(0)
            if node and not node.is_not_split():
                nodes.append(node.left)
                nodes.append(node.right)
            nodes_parsed.append(node)
        return nodes_parsed
    
    def get_nodes_at_depth(self, n):
        nodes = []
        for node in self.get_nodes():
            if node.depth == n:
                nodes.append(node)
        return nodes
    
    def get_avg_elli_area(self, n):
        elli_areas = []
        for node in self.get_nodes_at_depth(n):
            if node.split and isinstance(node.split, EllipseSplit):
                elli_areas.append(node.split.ellipse_area())
        return np.mean(elli_areas) if len(elli_areas)>0 else 0
    
    def get_split_ratios(self):
        total = 0
        ortho = 0
        diag = 0
        elli = 0
        for node in self.get_nodes():
            if node.split:
                total += 1
                if isinstance(node.split, OrthogonalSplit):
                    ortho += 1
                elif isinstance(node.split, DiagonalSplit):
                    diag += 1
                elif isinstance(node.split, EllipseSplit):
                    elli += 1
        if total == 0:
            return 0,0,0
        else: 
            return (ortho/total), (diag/total), (elli/total)
    
    def get_leafs(self):
        leafs = []
        for node in self.get_nodes():
            if node.is_leaf():
                leafs.append(node)
        return leafs

    def get_leafs_to_grow(self):
        leafs = []
        for node in self.get_nodes():
            if node.is_not_split() and not node.is_leaf():
                leafs.append(node)
        return leafs

    def get_leafs_not_split(self):
        leafs = []
        for node in self.get_nodes():
            if node.is_not_split():
                leafs.append(node)
        return leafs
    
    def set_metrics(self, metrics):
        self.metrics = metrics
    
    def get_metrics(self):
        return self.metrics
    
    def set_node_metrics(self, node_metrics):
        self.node_metrics = node_metrics
    
    def get_node_metrics(self):
        return self.node_metrics
    
    


class SplitGenerator:
    @staticmethod
    def generate_candidates(X):
        pass

class Split:
    def is_true(self, X):
        pass


