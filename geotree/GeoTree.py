"""
This is module contains the GeoTree method.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import resample, check_random_state
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from time import time
from . import _tree, _utils, _generators_numba, _generators_ga, _generators_sklearn
from ._generators_numba import OrthogonalSplitGenerator
from ._generators_ga import DiagonalSplitGenerator, EllipseSplitGenerator
import copy
import multiprocessing as mp
import joblib

FEATURE_THRESHOLD = 1e-7

params = {
        "num_gens":200, 
        "num_pop":100, 
        "eta": 0.2, 
        "tournsize": 10, 
        "alpha": 0.05, 
        "indpb":0.9, 
        "cxpb":0.9, 
        "mutpb":0.2,
        "hofsize":0,
        "regf":0
    }


GENERATORS = {
    "best": [
        _generators_sklearn.OrthogonalSplitGenerator,
        _generators_ga.DiagonalSplitGenerator,
        _generators_ga.EllipseSplitGenerator
    ]
}

class GeoTreeRegressor(BaseEstimator,RegressorMixin):
    """ The GeoTreeRegressor estimator.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are grown 
        until all leaves are pure or until all leaves contain less 
        than 2 samples.
    max_features : int, default=None
        The number of features to sample for each split. If None,
        then no sampling of features is performed.
    n_jobs : int, default=None
        The number of jobs to run in parallel. The 'fit' method is
        parallelized over nodes. If None, then 1 job is run, if -1 then
        all processors are used.
    random_state : int, RandomState instance or None, default=None
        Controlls randomness of estimator.

    
    Attributes
    ----------
    tree_ : Tree instance
        The underlying Tree object.
    generators_ : list of OrthogonalSplitGenerator, DiagonalSplitGenerator and EllipseSplitGenerator instances
        This list is used to generate the orthogonal, diagonal and ellipse candidate splits.
    n_features_in_ : int
        The number of features seen during training.
    features_ : list
        A list of feature names or indices used for training.
    feature_importances_ : ndarray of shape (n_features,)
        Impurity-based feature importances. Same as scikit-learn feature importances
        with an additional entry for the combination of geospatial features used in
        diagonal and ellipse splits.


    Examples
    --------
    >>> import GeoTreeRegressor
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = GeoTreeRegressor()
    >>> estimator.fit(X, y, geo_features=[0,1])
    GeoTreeRegressor()
    """
    def __init__(self, max_depth=None, max_features = None, n_jobs=None, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.is_fitted_ = False
        self.tree_ = None
        self.generators_ = None
        self.features_ = None
        self.n_features_in_ = -1
        self.random_state = random_state

    def fit_while(self, i, r):
        """Return True if stopping conditions for growing the tree are not satisfied, 
        otherwise False.

        Parameters
        ----------
        i : int 
            The current depth of the tree.
        r : int
            The maximum depth the tree should be grown.

        Returns
        -------
        ((i <= r) and a.any()) and (not b.all()) : Bool
            The current depth is smaller than the maximum depth, and at least one leaf 
            contains more than two samples, and not all leafs are pure or have constant features.
        

        """
        a = np.array([(len(leaf.idx) > 2) for leaf in self.tree_.get_leafs()])
        b = np.array([np.all(leaf.y==leaf.y[0]) or np.all(leaf.X==leaf.X[0]) for leaf in self.tree_.get_leafs()])
        if r>0:
            return ((i <= r) and a.any()) and (not b.all())
        else:
            return a.any() and (not b.all())

    def grow_leaf(self, leaf, geo_features, random_state):
        """Grow a leaf.

        Parameters
        ----------
        leaf : Node instance
            The leaf node that should be grown.
        geo_features : list
            The list of indices that indicate the columns of the geospatial features in the data.
        random_state: RandomState instance
            Controls the randomness.

        """
        if self.max_features:
            fs = copy.copy(self.features_)
            for gf in geo_features[1:]:
                fs.remove(gf)
            fs = resample(fs, n_samples=self.max_features, replace=False, random_state=random_state)
            gfs = []
            if any(item in geo_features for item in fs):
                fs = fs + geo_features
                fs = list(set(fs))
                gfs = geo_features
        else:
            fs = self.features_
            gfs = geo_features
        
        leaf.grow(generators=self.generators_,
                    features = fs,
                    geo_features=gfs,
                    n_jobs=self.n_jobs,
                    random_state=random_state
                    )

    def fit(self, X, y,
                generators = "best",
                ga_params = params,
                features = None, 
                geo_features = []):
        """Build a GeoTreeRegressor instance from the training set (X,y) with
        the given split generators, generator parameters, feature set, and geospatial features.        .


        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The training input samples.
        y : array, shape (n_samples,)
            The target values (real numbers).
        generators : string or list of generator instances
            The candidate split generators.
        ga_params : dict
            The parameters used for the GA-based generators
        features : list or None
            The features to use, if None then use all features.
        geo_features: list
            The list of indices that indicate the columns of the geospatial features in the data.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.is_fitted_ = True


        metrics_r = {}

        self.tree_ = _tree.Tree(X.astype('float32'),y.astype('float64'), self.max_depth)


        
        if generators == "best":
            self.generators_ = GENERATORS[generators]
            self.generators_ = [self.generators_[0], self.generators_[1](**ga_params), self.generators_[2](**ga_params)]
        
        self.features_ = features if features else list(range(X.shape[1]))
        self.max_features =  min(self.max_features, len(self.features_)) if self.max_features else None


        self.n_features_in_ = len(features) if features else X.shape[1]

        start = time()

        random_state = check_random_state(self.random_state)

        i=1
        while self.tree_.get_leafs_to_grow():

            self.tree_.curr_depth += 1

            leafs_to_grow = self.tree_.get_leafs_to_grow()


            joblib.Parallel(n_jobs=self.n_jobs,require='sharedmem')(
                joblib.delayed(self.grow_leaf)(leaf,
                                                geo_features, 
                                                random_state)
                for leaf in leafs_to_grow
            )

          

            end = time()

            # After each depth increase, calculate metrics
            metrics_r[i] = {}
            metrics_r[i] = _tree.calc_metrics(self.tree_.y, self.predict(self.tree_.X))
            metrics_r[i]['time'] = end-start
            metrics_r[i]['leaves'] = len(self.tree_.get_leafs_not_split())
            metrics_r[i]['true_leaves'] = self.tree_.n_leaves
            metrics_r[i]['elli_area'] = self.tree_.get_avg_elli_area(i)
            metrics_r[i]['ortho_ratio'],\
                metrics_r[i]['diag_ratio'],\
                    metrics_r[i]['elli_ratio'] = self.tree_.get_split_ratios()
            
            self.tree_.set_metrics(metrics_r)

            i+=1
        
        return self

    def predict(self, X):
        """ Predict value for samples in X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array, shape (n_samples,)
            Returns an array of predictions.
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        yhat = np.zeros(len(X))

        nodes_to_expand = [(self.tree_.root, np.arange(0, len(X)))]

        while nodes_to_expand:
            expand_next = []
            for node_to_expand, idx in nodes_to_expand:
                if node_to_expand.is_not_split():
                    yhat[idx] = node_to_expand.yhat
                else:
                    idx_left, idx_right = node_to_expand.get_left_right_idx(X[idx], idx)
                    expand_next.append((node_to_expand.left, idx_left))
                    expand_next.append((node_to_expand.right, idx_right))
            nodes_to_expand = expand_next
        
        return yhat

    def predict_at_depth(self, X, n):
        """ Predict value for samples in X using the regression tree
        at depth n.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input samples.
        n : int
            The depth at which to predict.

        Returns
        -------
        y : array, shape (n_samples,)
            Returns an array of predictions.
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        yhat = np.zeros(len(X))

        nodes_to_expand = [(self.tree_.root, np.arange(0, len(X)))]

        while nodes_to_expand:
            expand_next = []
            for node_to_expand, idx in nodes_to_expand:
                if node_to_expand.is_leaf() or (node_to_expand.depth==n):
                    yhat[idx] = node_to_expand.yhat
                else:
                    idx_left, idx_right = node_to_expand.get_left_right_idx(X[idx], idx)
                    expand_next.append((node_to_expand.left, idx_left))
                    expand_next.append((node_to_expand.right, idx_right))
            nodes_to_expand = expand_next
        
        return yhat

    def get_depth(self):
        """Return the depth of the decision tree.

        Returns
        -------
        self.tree_.curr_depth : int
            The current depth of the tree.
        """
        return self.tree_.curr_depth

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        len(self.tree_.get_leafs()) : int
            The number of leaves of the tree.
        """
        return len(self.tree_.get_leafs())

    @property
    def feature_importances_(self, normalize=True):
        """Return the feature importances.
        The importance of a feature is computed as the (normalized) total
        reduction of the MSE due to all splits based on that feature.
        
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total reduction of MSE by feature.
        """
        importances = np.zeros(self.n_features_in_+1, dtype=np.float64)
        for node in self.tree_.get_nodes():
            if node.split:
                if hasattr(node.split,'feat'):
                    importances[node.split.feat] += node.gain
                else:
                    importances[-1] += node.gain
        importances /= self.tree_.X.shape[0]

        if normalize:
            normalizer = np.sum(importances)
            if normalizer > 0.0:
                importances /= normalizer
        return importances

    def print_splits_depth_first(self, node):
        """Helper function to recursively print the tree

        Parameters
        ----------
        node : Node instance
            The node to print.
        """
        if node.split:
            print(' '*node.depth + str(node.split) + f', squared_error= {node.mse}, samples= {len(node.idx)}, value= {node.yhat}')
            self.print_splits_depth_first(node.left)
            self.print_splits_depth_first(node.right)
        else:
            print(' '*node.depth +f"Leaf: pred {node.yhat}")

    def plot_tree(self):
        """Print the tree.
        
        """
        self.print_splits_depth_first(self.tree_.root)

