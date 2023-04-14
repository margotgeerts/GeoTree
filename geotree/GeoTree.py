"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import resample, check_random_state
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from time import time
from . import _tree, _utils, _generators_numba, _generators_ga, _generators_sklearn, _generators_random
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
        "regf":0,
        "eval_ratio":1.
    }


GENERATORS = {
    "best": [
        _generators_sklearn.OrthogonalSplitGenerator,
        _generators_ga.DiagonalSplitGenerator,
        _generators_ga.EllipseSplitGenerator
    ]
}

class GeoTreeRegressor(BaseEstimator,RegressorMixin):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> import GeoTreeRegressor
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = GeoTreeRegressor()
    >>> estimator.fit(X, y)
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
        a = np.array([(len(leaf.idx) > 2) for leaf in self.tree_.get_leafs()])
        b = np.array([np.all(leaf.y==leaf.y[0]) or np.all(leaf.X==leaf.X[0]) for leaf in self.tree_.get_leafs()])
        if r>0:
            return ((i <= r) and a.any()) and (not b.all())
        else:
            return a.any() and (not b.all())

    def grow_leaf(self, leaf, geo_features, random_state, CONTINUE_NONE, n_RESTART, geosplits, min_area, min_gain):

        if self.max_features:
            #print(self.features_, geo_features)
            fs = copy.copy(self.features_)
            for gf in geo_features[1:]:
                fs.remove(gf)
            fs = resample(fs, n_samples=self.max_features, replace=False, random_state=random_state)
            gfs = []
            if any(item in geo_features for item in fs):
                fs = fs + geo_features
                fs = list(set(fs))
                gfs = geo_features
            #print(fs, gfs)
        else:
            fs = self.features_
            gfs = geo_features
        
        leaf.grow(generators=self.generators_,
                    features = fs,
                    geo_features=gfs,
                    CONTINUE_NONE=CONTINUE_NONE,
                    n_RESTART=n_RESTART,
                    geosplits=geosplits,
                    min_area=min_area,
                    min_gain=min_gain,
                    n_jobs=self.n_jobs,
                    random_state=random_state
                    )

    def fit(self, X, y,
                generators = "best",
                ga_params = params,
                random_params = random_params,
                features = None, 
                geo_features = [],
                CONTINUE_NONE = True,
                n_RESTART = 0,
                X_test = None, y_test = None,
                save_fig = False,
                geosplits = None, min_area = 0.0,
                early_stopping = None,
                min_gain = 0.0):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.is_fitted_ = True

        trees = []
        metrics_r = {}
        #node_metrics = {}
        #n=1
        self.tree_ = _tree.Tree(X.astype('float32'),y.astype('float64'), self.max_depth)
        X_test = X_test.astype('float32') if X_test is not None else None
        y_test = y_test.astype('float64') if y_test is not None else None

        self.generators_ = GENERATORS[generators]
        if generators == "best":
            self.generators_ = [self.generators_[0], self.generators_[1](**ga_params), self.generators_[2](**ga_params)]
        if generators == "random":
            self.generators_ = [gen(k) for (gen, k) in zip(self.generators_, random_params)]

        self.features_ = features if features else list(range(X.shape[1]))
        self.max_features =  min(self.max_features, len(self.features_)) if self.max_features else None


        self.n_features_in_ = len(features) if features else X.shape[1]

        start = time()

        random_state = check_random_state(self.random_state)

        print(self.tree_.max_depth)

        i=1
        while self.tree_.get_leafs_to_grow():

            self.tree_.curr_depth += 1

            leafs_to_grow = self.tree_.get_leafs_to_grow()


            joblib.Parallel(n_jobs=self.n_jobs,require='sharedmem')(
                joblib.delayed(self.grow_leaf)(leaf,
                                                geo_features, 
                                                random_state, 
                                                CONTINUE_NONE, 
                                                n_RESTART, 
                                                geosplits, 
                                                min_area, 
                                                min_gain)
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
            
            if X_test is not None:
                
                metrics_test = _tree.calc_metrics(y_test, self.predict(X_test), "test")
                metrics_r[i].update(metrics_test)
                
                if early_stopping:
                    
                    if (i>0) and (trees[-1][0] < metrics_test['maetest']):
                        self.tree_ = trees[-1][1]
                        break
                    else:
                        t = copy.deepcopy(self.tree_)
                        trees.append([metrics_test['maetest'],t])
            
            print(str(i),self.tree_.n_leaves,metrics_r[i])
            self.tree_.set_metrics(metrics_r)
            i+=1
            
            if save_fig:
                _utils.plot_decision_boundary(self.tree_.X,self.tree_.y,self.predict)
                plt.savefig(save_fig+f"_{i}.png")
        
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
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
        return self.tree_.curr_depth

    def get_n_leaves(self):
        return len(self.tree_.get_leafs())

    @property
    def feature_importances_(self, normalize=True):
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
        if node.split:
            print(' '*node.depth + str(node.split) + f', squared_error= {node.mse}, samples= {len(node.idx)}, value= {node.yhat}')
            self.print_splits_depth_first(node.left)
            self.print_splits_depth_first(node.right)
        else:
            print(' '*node.depth +f"Leaf: pred {node.yhat}")

    def plot_tree(self):
        self.print_splits_depth_first(self.tree_.root)

