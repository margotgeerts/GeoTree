import numpy as np
from ._splits import *
from numba import jit, prange

FEATURE_THRESHOLD = 1e-7


@jit(nopython=True, nogil=True, fastmath=True)
def mse(y, y_hat):
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
    diffsq / y.shape[0] : float
    """
    diffsq = 0
    for k in range(y.shape[0]):
        diffsq += (y[k] - y_hat)**2
    return diffsq / y.shape[0]

@jit(nopython=True, nogil=True, fastmath=True)
def euclidean(p1x, p1y, p2x, p2y):
    """
    Calculate the Euclidean distance between two points in a two-dimensional space.
    
    Parameters
    ----------
    p1x : float
        The x-coordinate of the first point, p1
    p1y : float
        The y-coordinate of the first point, p1
    p2x : float
        The x-coordinate of the second point, p2
    p2y : float
        The y-coordinate of the second point, p2

    Returns
    -------
    ((p1x - p2x)**2 + (p1y - p2y)**2)**(.5): float
        The Euclidean distance between two points (p1x, p1y) and (p2x, p2y).
    """
    return ((p1x - p2x)**2 + (p1y - p2y)**2)**(.5)

@jit(nopython=True, nogil=True, fastmath=True)
def calc_gain(y, tidx, parent_mse, n):
    """
    Calculate the potential gain in mean squared error (MSE) from splitting a decision
    tree node based on the given array.
    
    Parameters
    ----------
    y : np.array, shape (n_samples,)
        The target variable.
    tidx : array-like
        Boolean array indicating the samples that satisfy the split
        and the samples that don't.
    parent_mse : float
        The mean squared error of the parent node before the split.
    n : int
        The total number of samples in the dataset.

    Returns
    -------
    gain : np.float64
        The gain in mean squared error (MSE) resulting from the candidate split. 
    """
    num_obs = y.shape[0]
    left_y, right_y = y[~tidx], y[tidx]
    num_left = left_y.shape[0]
    num_right = right_y.shape[0]
    if not num_left or not num_right:
        return 0
    left_mse = mse(left_y, np.mean(left_y))
    right_mse = mse(right_y, np.mean(right_y))
    left_ratio = (num_left / num_obs)
    right_ratio = 1 - left_ratio
    weight = num_obs / n
    gain = weight * (parent_mse - (left_mse * left_ratio) - (right_mse * right_ratio))
    return np.float64(gain)

@jit(nopython=True, nogil=True, fastmath=True)
def ellipse_is_true(X, f1, f2, focal1x, focal1y, focal2x, focal2y, dist):
    """
    Return whether the given points are on or outside of the ellipse.
    
    Parameters
    ----------
    X : array
        The input samples.
    f1 : int
        The index of the first feature column.
    f2 : int
        The index of the second feature column.
    focal1x : float
        The x-coordinate of the first focal point of the ellipse.
    focal1y : float
        The y-coordinate of the first focal point of the ellipse.
    focal2x : float
        The x-coordinate of the second focal point of the ellipse.
    focal2y : float
        The y-coordinate of the second focal point of the ellipse.
    dist : float
        The constant distance from any point on the ellipse
        to the two focal points.
    bbox : float
        The spatial extent of the input data.

    Returns
    -------
    dist_1 + dist_2 >= dist: boolean array 
        True if given point is on or outside of the ellipse, otherwise False.
    """
    dist_1 = np.sqrt((X[:, f1] - focal1x)**2 + (X[:, f2] - focal1y)**2)
    dist_2 = np.sqrt((X[:, f1] - focal2x)**2 + (X[:, f2] - focal2y)**2)
    return dist_1 + dist_2 >= dist

@jit(nopython=True, nogil=True, fastmath=True)
def diagonal_is_true(X, f1, f2, intercept, slope):
    """
    Return whether the given points are on or above the diagonal.
    
    Parameters
    ----------
    X : array
        The input samples.
    f1 : int
        The index of the first feature column.
    f2 : int
        The index of the second feature column.
    intercept : float
        The intercept of the diagonal.
    slope : float
        The slope of the diagonal.

    Returns
    -------
    X[:,f2] >= intercept + slope * X[:,f1]: boolean array 
        True if given point is on or above the diagonal, otherwise False.
    """
    return X[:,f2] >= intercept + slope * X[:,f1]

@jit(nopython=True, nogil=True, fastmath=True)
def orthogonal_is_true(X, f, split):
    """
    Return whether the given points are on or to the right of the orthogonal.
    
    Parameters
    ----------
    X : array
        The input samples.
    f : int
        The index of the feature column.
    split : float
        The split threshold.

    Returns
    -------
    X[:,f] >= split: boolean array 
        True if given point is on or to the right of the orthogonal, otherwise False.
    """
    return X[:,f] >= split

@jit(nopython=True, nogil=True, fastmath=True, cache=True, parallel=False)
def orthogonal_evaluate_candidates(X, y, parent_mse, features, n):
    """
    Evaluate all possible orthogonal candidate splits.

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
    Returns
    -------
    best_mse_gain, best_split : float, tuple
        The mean squared error (MSE) gain and the corresponding orthogonal split 
        resulting in the highest gain.
    """
    best_mse_gain, best_split = 0.0, (None, None)
    for f in features:
        unique_sorted = np.sort(np.unique(X[:,f]))
        for i in range(1, len(unique_sorted)):
            if unique_sorted[i] <= unique_sorted[i-1] + FEATURE_THRESHOLD:
                continue
            split = (unique_sorted[i-1] + unique_sorted[i]) / 2
            if (split == unique_sorted[i] or
                split == np.inf or split == -np.inf):
                split = unique_sorted[i-1]
            tidx = orthogonal_is_true(X, f, split)
            gain = calc_gain(y, tidx, parent_mse,n)
            if gain >= best_mse_gain:
                best_mse_gain = gain
                best_split = (f, split)
    return best_mse_gain, best_split

@jit(nopython=True, nogil=True, fastmath=True)
def diagonal_evaluate_candidates(X, y, parent_mse, n):
    """
    Evaluate all possible diagonal candidate splits.

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
    Returns
    -------
    best_mse_gain, best_split : float, tuple
        The mean squared error (MSE) gain and the corresponding diagonal split 
        resulting in the highest gain.
    """
    best_mse_gain, best_split = 0.0, (None, None, None, None)
    for f1 in range(X.shape[1]):
        for f2 in range(f1+1, X.shape[1]):
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    slope = (X[j,f2] - X[i,f2]) / (X[j,f1] - X[i,f1])
                    x1 = X[i,f1] + (X[j,f1] - X[i,f1]) / 2
                    x2 = slope * (x1 - X[i,f1]) + X[i,f2]
                    islope = -1/slope
                    iint = x2 - islope * x1
                    tidx = diagonal_is_true(X, f1, f2, iint, islope)
                    gain = calc_gain(y, tidx, parent_mse,n)
                    if gain >= best_mse_gain:
                        best_mse_gain = gain
                        best_split = (f1, f2, iint, islope)
    return best_mse_gain, best_split

@jit(nopython=True, nogil=True, fastmath=True)
def ellipse_evaluate_candidates(X, y, parent_mse, n):
    """
    Evaluate all possible ellipse candidate splits.

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
    Returns
    -------
    best_mse_gain, best_split : float, tuple
        The mean squared error (MSE) gain and the corresponding ellipse split
        resulting in the highest gain.
    """
    best_mse_gain, best_split = 0.0, (None, None, None, None, None, None, None)
    for f1 in range(X.shape[1]):
        for f2 in range(f1+1, X.shape[1]):
            f12 = np.array([f1, f2])
            for i in range(len(X)):
                xi = X[i,:]
                for j in range(i, len(X)):
                    xj = X[j,:]
                    xh = (xi[f12] + xj[f12]) / 2.0
                    for k in range(len(X)):
                        if k == i or k == j:
                            continue
                        xk = X[k,:]
                        distance = euclidean(xk[f1], xk[f2], xh[f1], xh[f2])
                        distanceij = euclidean(xi[f1], xi[f2], xj[f1], xj[f2])
                        if distance < distanceij:
                            continue
                        tidx = ellipse_is_true(X, f1, f2, xi[f1], xi[f2], xj[f1], xj[f2], distance)
                        gain = calc_gain(y, tidx, parent_mse,n)
                        if gain >= best_mse_gain:
                            best_mse_gain = gain
                            best_split = (f1, f2, xi[f1], xi[f2], xj[f1], xj[f2], distance)
    return best_mse_gain, best_split

class OrthogonalSplitGenerator:
    """The OrthogonalSplitGenerator class.
    """       
    @staticmethod
    def generate_candidates(X, y, parent_mse, features, geo_features, n, bbox, random_state):
        """
        Generate orthogonal candidate splits using brute force.
        
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
        gain, (f, split) = orthogonal_evaluate_candidates(X, y, parent_mse, features, n)
        yield OrthogonalSplit(f, split)

class DiagonalSplitGenerator:
    """The DiagonalSplitGenerator class.
    """
    @staticmethod
    def generate_candidates(X, y, parent_mse, features, geo_features, n, bbox, random_state):
        """
        Generate diagonal candidate splits using brute force.
        
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
        A generator that returns DiagonalSplit objects.
        """
        gain, (f1, f2, iint, islope) = diagonal_evaluate_candidates(X, y, parent_mse, n)
        yield DiagonalSplit(f1, f2, iint, islope)

class EllipseSplitGenerator:
    """The EllipseSplitGenerator class.
    """
    @staticmethod
    def generate_candidates(X, y, parent_mse, features, geo_features, n, bbox, random_state):
        """
        Generate ellipse candidate splits using brute force.
        
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
        A generator that returns EllipseSplit objects.
        """
        gain, (f1, f2, xif1, xif2, xjf1, xjf2, distance) = ellipse_evaluate_candidates(X, y, parent_mse, n)
        yield EllipseSplit(f1, f2, np.array([xif1, xif2]), np.array([xjf1, xjf2]), distance, bbox)




