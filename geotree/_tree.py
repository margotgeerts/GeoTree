import numpy as np
import matplotlib.pyplot as plt
from ._splits import *
from time import time
import joblib
import warnings

DEBUG_SPLITS = False
EPSILON = np.finfo('double').eps


def mse(y_true, y_pred):
    """
    Calculate the mean squared error between the true and predicted values of a dataset.
    
    Parameters
    ----------
    y_true : array, shape (n,)
        The true values of the target variable.
    y_pred : {array, float, int}
        The predicted values of the target variable. An array of shape (n,) or a float/int.

    Returns
    -------
    np.sum((y_pred - y_true)**2) / len(y_true) : float
    """
    return np.sum((y_pred - y_true)**2) / len(y_true)

def rmse(y_true, y_pred):
    """
    Calculate the root mean squared error (RMSE) between the true and predicted values.
    
    Parameters
    ----------
    y_true : array, shape (n,)
        The true values of the target variable.
    y_pred : {array, float, int}
        The predicted values of the target variable. An array of shape (n,) or a float/int.

    Returns
    -------
    np.sqrt(mse(y_true, y_pred)) : float
    """
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    """
    Calculate the mean absolute error between the true and predicted values.
    
    Parameters
    ----------
    y_true : array, shape (n,)
        The true values of the target variable.
    y_pred : {array, float, int}
        The predicted values of the target variable. An array of shape (n,) or a float/int.

    Returns
    -------
    np.mean(np.abs(y_pred - y_true)) : float
    """
    return np.mean(np.abs(y_pred - y_true))

def mape(y_true,y_pred):
    """
    Calculate the mean absolute percentage error (MAPE) between the true and predicted
    values of a dataset.
    
    Parameters
    ----------
    y_true : array, shape (n,)
        The true values of the target variable.
    y_pred : {array, float, int}
        The predicted values of the target variable. An array of shape (n,) or a float/int.

    Returns
    -------
    np.mean(np.abs((y_pred-y_true))/np.maximum(np.abs(y_true),EPSILON)) : float
    """
    return np.mean(np.abs((y_pred-y_true))/np.maximum(np.abs(y_true),EPSILON))

def r2_score(y_true,y_pred):
    """
    Calculate the R-squared score between the true and predicted values of a regression
    model.
    
    Parameters
    ----------
    y_true : array, shape (n,)
        The true values of the target variable.
    y_pred : {array, float, int}
        The predicted values of the target variable. An array of shape (n,) or a float/int.

    Returns
    -------
    r2 : float
    """
    y_bar = np.sum(y_true) / len(y_true)
    ssr = np.sum((y_pred-y_true)**2)
    sst = np.sum((y_bar-y_true)**2)
    return 1 - (ssr/sst)

def calc_metrics(y_true, y_pred, string=""):
    """
    Calculate and return various metrics such as RMSE, MAE, MAPE, and R2 score for given
    true and predicted values.
    
    Parameters
    ----------
    y_true : array, shape (n,)
        The true values of the target variable.
    y_pred : {array, float, int}
        The predicted values of the target variable. An array of shape (n,) or a float/int.
    string : string
        Optional parameter for a string that can be added to the metric names to
        differentiate between different sets of predictions.

    Returns
    -------
    metrics : dict
        A dictionary containing the calculated metrics for the input true and predicted values,
        with keys indicating the name of the metric.
    """
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


def evaluate_split(candidate_split, X, y, parent_mse, n):
    """
    Evaluates the potential gain in mean squared error (MSE) from splitting a decision
    tree node based on the given candidate split.
    
    Parameters
    ----------
    candidate_split : Split object
        Candidate_split is Split object that represents a potential split in a decision
        tree.
    X : np.array, shape (n_samples, n_features)
        A numpy array containing the input samples. 
    y : np.array, shape (n_samples,)
        The target variable.
    parent_mse : float
        The mean squared error of the parent node before the split.
    n : int
        The total number of samples in the dataset.

    Returns
    -------
    gain : np.float64
        The gain in mean squared error (MSE) resulting from the candidate split. 
    """
    if candidate_split is None:
        return 0
    tidx = candidate_split.is_true(X)
    left_y, right_y = y[~tidx], y[tidx]
    if not len(left_y) or not len(right_y):
        return 0 # Zou niet mogen voorvallen
    left_mse = mse(left_y, np.mean(left_y))
    right_mse = mse(right_y, np.mean(right_y))
    n_t = len(y)
    left_ratio = len(left_y) / n_t
    right_ratio = len(right_y) / n_t
    weight = n_t / n
    gain = weight * (parent_mse - (left_mse * left_ratio) - (right_mse * right_ratio))
    return np.float64(gain)


def evaluate_splits(generator, X, y, parent_mse, features, geo_features, n, bbox, random_state):
    """
    Evaluate candidate splits generated by the given generators.
    
    Parameters
    ----------
    generator : {OrthogonalSplitGenerator, DiagonalSplitGenerator, EllipseSplitGenerator} 
        An object that generates candidate splits for a decision tree.
    X : np.array, shape (n_samples, n_features)
        A numpy array containing the input samples. 
    y : np.array, shape (n_samples,)
        The target variable.
    parent_mse : float
        The mean squared error of the parent node before the split.
    n : int
        The total number of samples in the dataset.
    features : array-like
        A list of feature names or indices.
    geo_features : array-like
        A list of indices indicating a set of geospatial features.
    bbox : float
        Bounding box indiciating the spatial extent of the input data.
    random_state: RandomState instance
        Controls the randomness.
    
    Returns
    -------
    best_mse_gain, best_split : float, Split object
        The mean squared error (MSE) gain and the corresponding Split object resulting
        in the highest gain.
    """
    best_mse_gain, best_split = 0.0, None
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
        if gain >= best_mse_gain:
            best_mse_gain = gain
            best_split = candidate_split
    return best_mse_gain, best_split


# The Node class represents a node in a decision tree with attributes such as index, depth, parent,
# left and right children, split, and gain.
class Node:
    """ The Node class.

    Parameters
    ----------
    tree : Tree object
        The Tree object that the node belongs to.
    idx : array
        The indices of the input samples sorted into this node.
    depth : int
        The depth of the node in a tree. The root node has a
        depth of 0, and each subsequent level of nodes has a greater depth.
    parent : {Node object, None}
        The parent node. It is None for the root node.

    Attributes
    ----------
    left : Node object
        The left child node.
    right : Node object
        The right child node.
    split : Split object
        The split object that splits this node.
    gain : float
        The gain in MSE resulting from the split.
    """
    def __init__(self, tree, idx, depth, parent=None):
        self.tree = tree
        self.idx = idx
        self.depth = depth
        self.parent = parent
        self.left = None
        self.right = None
        self.split = None
        self.gain = None
        

    @property
    def X(self):
        """
        Return the input samples sorted into this node.
        """
        return self.tree.X[self.idx]
    
    @property
    def y(self):
        """
        Return the target values corresponding to the input
        samples sorted into this node.
        """
        return self.tree.y[self.idx]

    @property
    def yhat(self):
        """
        Return the predicted value in this node, which is the average of
        the target values sorted into this node.
        """
        return np.mean(self.tree.y[self.idx]) if (isinstance(self.tree.y,(list,np.ndarray)) and  len(self.tree.y)>1) else self.tree.y
    
    @property
    def mse(self):
        """
        Calculate the impurity, or the mean squared error between the predicted value (yhat) and the
        actual values (y).
        """
        return mse(self.y, self.yhat)

    def is_not_split(self):
        """
        Check if a binary tree node has no left or right child nodes.
        """
        return self.left is None and self.right is None

    def is_leaf(self):
        """
        Check if the current node is a leaf node. 
        """
        return ((self.tree.max_depth and self.depth >= self.tree.max_depth) or # max depth reached
                  len(self.idx) < 2 or # only one sample in node
                  ((self.gain is not None) and self.gain <= self.min_gain) or # tried to find split but gain is 0
                  self.mse <= EPSILON or    # mse is 0
                  np.all(self.X==self.X[geo_features[0]])) # all features are constant
    
    
    def best_split(self, generators, features, geo_features, n_jobs, random_state):
        """
        Find the best split for a decision tree using generators and parallel processing.
        
        Parameters
        ----------
        generators : list
            A list of generator objects that generate candidate splits for the decision
            tree.
        features : array-like
            A list of feature names or indices.
        geo_features : array-like
            A list of indices indicating a set of geospatial features.
        n_jobs : int
            The number of jobs to run in parallel. The method is
            parallelized over generators. If None, then 1 job is run, if -1 then
            all processors are used.
        random_state : RandomState instance
            Controlls the randomness.

        Returns
        -------
        best_mse_gain, best_split : float, Split object
            The mean squared error (MSE) gain and the corresponding Split object resulting
            in the highest gain.
        """
        assert generators

        if np.all(self.X[:,features]==self.X[0,features]):
            return best_mse_gain, best_split

        gains, splits = zip(*joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(evaluate_splits)(generator,
                                                                                                self.X, self.y, 
                                                                                                self.mse,
                                                                                                features,
                                                                                                geo_features, 
                                                                                                len(self.tree.y), 
                                                                                                self.tree.bbox,
                                                                                                random_state) for generator in generators))
        
        best_mse_gain = np.max(gains)
        best_split = splits[np.argmax(gains)]

        if DEBUG_SPLITS:
            print(TREE_BALANCE_BIAS, best_mse_gain, best_split, len(self.tree.get_leafs()))

        return best_mse_gain, best_split
    
    def get_left_right_idx(self, X=None, idx=None):
        """
        Return the indices of the left and right child nodes of a decision tree node.
        
        Parameters
        ----------
        X : np.array, shape (n_samples, n_features), default = None
            A numpy array containing the input samples. 
        idx : array, default = None
            An array of indices to split.

        Returns
        -------
        idx_left, idx_right : array, array
            The indices of the left and right child nodes of a decision tree
            node, given the input data and indices of the current node.
        """
        if X is None:
            X = self.X
            idx = self.idx
        if self.split is None:
            return None, None
        tidx = self.split.is_true(X)
        idx_left = idx[np.where(~tidx)[geo_features[0]]]
        idx_right = idx[np.where(tidx)[geo_features[0]]]
        return idx_left, idx_right

    def grow(self, generators, features, geo_features, n_jobs=None, random_state=None):
        """
        Grow the tree by expanding the current node based on the best mean squared
        error gain.
        
        Parameters
        ----------
        generators : list
            A list of generator objects that generate candidate splits for the decision
            tree.
        features : array-like
            A list of feature names or indices.
        geo_features : array-like
            A list of indices indicating a set of geospatial features.
        n_jobs : int
            The number of jobs to run in parallel. The method is
            parallelized over generators. If None, then 1 job is run, if -1 then
            all processors are used.
        random_state : RandomState instance
            Controlls the randomness.
        
        """
        best_mse_gain, best_split = self.best_split(generators, features, geo_features, n_jobs, random_state)
        self.split = best_split
        self.gain = best_mse_gain
        idx_left, idx_right = self.get_left_right_idx()
        if idx_left is not None:
            self.left = Node(self.tree, idx_left, self.depth+1, self)
        if idx_right is not None:
            self.right = Node(self.tree, idx_right, self.depth+1, self)

class Tree:
    """ The Tree class.

    Parameters
    ----------
     X : array, shape (n_samples, n_features)
        The training input samples.
    y : array, shape (n_samples,)
        The target values (real numbers).
    max_depth : int
        The maximum depth of the tree. 

    Attributes
    ----------
    root : Node object
        The root node.
    metrics : dict
        A dictionary to retain error metrics.
    curr_depth : int
        The current depth of the tree.

    """
    def __init__(self, X, y, max_depth):
        self.root = Node(self, np.arange(0, len(X)), 0)
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.metrics = None
        self.curr_depth = 0
        
    @property
    def bbox(self, geo_features=[0,1]):
        """
        Calculate the spatial extent of the input data.
        
        Parameters
        ----------
        geo_features : array-like, default = [0,1]
            A list of indices indicating a set of geospatial features. The default is the first two features.

        Returns
        -------
        bbox : float
        """
        if self.X.shape[1] > 1:
            return (max(self.X[:,geo_features[0]]) - min(self.X[:,geo_features[0]])) * (max(self.X[:,geo_features[1]]) - min(self.X[:,geo_features[1]]))
        else:
            return 0.0

    @property
    def n_leaves(self):
        """
        Return the number of leaves in the tree.
        """
        return len(self.get_leafs())

    @property
    def node_count(self):
        """
        Return the number of nodes in the tree.
        """
        return len(self.get_nodes())

    def get_nodes(self):
        """
        Return a list of all nodes in the tree.
        """
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
        """
        Return a list of nodes at a specified depth in a tree.
        
        Parameters
        ----------
        n : int
            The depth

        Returns
        -------
        nodes : list
            List of node objects.
        """
        nodes = []
        for node in self.get_nodes():
            if node.depth == n:
                nodes.append(node)
        return nodes
    
    def get_avg_elli_area(self, n):
        """
        Calculate the average area of ellipses at a given depth in a tree structure.
        
        Parameters
        ----------
        n : int
            The depth

        Returns
        -------
        np.mean(elli_areas) : float
            The average area of ellipses.
        """
        elli_areas = []
        for node in self.get_nodes_at_depth(n):
            if node.split and isinstance(node.split, EllipseSplit):
                elli_areas.append(node.split.ellipse_area())
        return np.mean(elli_areas) if len(elli_areas)>0 else 0
    
    def get_split_ratios(self):
        """
        Calculate the ratio of different types of splits in the tree.
        """
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
        """
        Return a list of leaf nodes in a tree.
        """
        leafs = []
        for node in self.get_nodes():
            if node.is_leaf():
                leafs.append(node)
        return leafs

    def get_leafs_to_grow(self):
        """
        Return a list of nodes that are not split and do not satisfy the conditions
        of a leaf node.
        """
        leafs = []
        for node in self.get_nodes():
            if node.is_not_split() and not node.is_leaf():
                leafs.append(node)
        return leafs

    def get_leafs_not_split(self):
        """
        Return a list of nodes that have not been split in the tree.
        """
        leafs = []
        for node in self.get_nodes():
            if node.is_not_split():
                leafs.append(node)
        return leafs
    
    def set_metrics(self, metrics):
        """
        Set the value of the "metrics" attribute of the tree to the input parameter
        "metrics".
        
        Parameters
        ----------
        metrics : dict
            A dictionary containing the error metrics.

        """
        self.metrics = metrics
    
    def get_metrics(self):
        """
        Return the metrics of the tree.
        """
        return self.metrics
    
    


class SplitGenerator:
    """The base class for split generators.
    """
    @staticmethod
    def generate_candidates(X):
        pass

class Split:
    """The base class for splits.
    """
    def is_true(self, X):
        pass


