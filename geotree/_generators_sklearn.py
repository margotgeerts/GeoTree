import numpy as np
from ._splits import *
from sklearn.tree import DecisionTreeRegressor


class OrthogonalSplitGenerator:
    """The OrthogonalSplitGenerator class for sklearn-based splits.
    """
    def generate_candidates(X, y, parent_mse, features, geo_features, n, bbox, random_state):
        """
        Generate orthogonal candidate splits using scikit-learn decision trees.
        
        Parameters
        ----------
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
            Bounding box indiciating the spatial extent of the input data. (Unused here)
        random_state : RandomState instance
            Unused here
        
        Returns
        -------
        A generator that returns OrthogonalSplit objects.
        """
        dt = DecisionTreeRegressor(max_depth=1, random_state=random_state)
        dt.fit(X[:,features],y)
        f = features[dt.tree_.feature[0]] if dt.tree_.feature[0] >= 0 else np.nan
        split = dt.tree_.threshold[0] if dt.tree_.feature[0] >= 0 else np.nan
        orthosplit = OrthogonalSplit(f, split) if ~np.isnan(f) else None
        yield orthosplit
