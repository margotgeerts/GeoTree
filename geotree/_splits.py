import numpy as np

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

class EllipseSplit:
    """ The EllipseSplit class.
        
        Parameters
        ----------
        feat1 : int
            The index of the first feature column.
        feat2 : int
            The index of the second feature column.
        focal1 : np.array
            The first focal point of the ellipse.
        focal2 : np.array
            The second focal point of the ellipse.
        dist : float
            The constant distance from any point on the ellipse
            to the two focal points.
        bbox : float
            The spatial extent of the input data.
        regf : float, default = 0
            A regularization parameter to punish for large ellipses.
        """
    def __init__(self, feat1, feat2, focal1, focal2, dist, bbox, regf=0):
        self.feat1 = feat1
        self.feat2 = feat2
        self.focal1 = np.copy(focal1).astype(np.float32)
        self.focal2 = np.copy(focal2).astype(np.float32)
        self.dist = dist
        self.regf = regf
        self.bbox = bbox

    def __str__(self):
        """
        Return a string representation of the ellipse.
        """
        return f"d(X[{self.feat1}], {self.focal1}) + d(X[{self.feat2}], {self.focal2}) >= {self.dist}"

    def ellipse_area(self):
        """
        Return the area of the ellipse.
        """
        b = self.dist
        distf1f2 = euclidean(self.focal1[0],self.focal1[1],self.focal2[0],self.focal2[1])
        a = np.sqrt((b**2) + ((distf1f2/2)**2))
        return (np.pi * a * b) / self.bbox
    
    def is_true(self, X):
        """
        Return whether the given points are on or outside of the ellipse.
        
        Parameters
        ----------
        X : array
            The input samples.

        Returns
        -------
        dist_1 + dist_2 >= self.dist: boolean array 
            True if given point is on or outside of the ellipse, otherwise False.
        """
        dist_1 = np.sqrt((X[:, self.feat1] - self.focal1[0])**2 + (X[:, self.feat2] - self.focal1[1])**2)
        dist_2 = np.sqrt((X[:, self.feat1] - self.focal2[0])**2 + (X[:, self.feat2] - self.focal2[1])**2)
        return dist_1 + dist_2 >= self.dist
    
    def evaluate_ellipse(self, gain, n_t, n):
        """
        Calculate the gain of an ellipse and applies regularization if necessary.
        
        Parameters
        ----------
        gain : float
            Improvement in MSE.
        n_t: int
            The number of samples sorted in the current tree node.
        n : int
            The total number of input samples.

        Returns
        -------
        gain: float
            (Regularized) gain of the ellipse split.
        """
        if self.regf > 0:
          regularisation = ((n_t/n) * self.regf * self.ellipse_area())
          if not gain or regularisation >= gain:
              return 0.0
          return (gain - regularisation)
        
        return gain
          

class DiagonalSplit:
    """ The DiagonalSplit class.
    
    Parameters
    ----------
    feat1 : int
        The index of the first feature column.
    feat2 : int
        The index of the second feature column.
    intercept : float
        The intercept of the diagonal.
    slope : float
        The slope of the diagonal.
    """
    def __init__(self, feat1, feat2, intercept, slope):
        self.feat1 = feat1
        self.feat2 = feat2
        self.intercept = intercept
        self.slope = slope

    def __str__(self):
        """
        Return a string representation of the diagonal.
        """
        return f"X[{self.feat2}] >= {self.intercept} + {self.slope} * X[{self.feat1}]"

    def is_true(self, X):
        """
        Return whether the given points are on or above the diagonal.
        
        Parameters
        ----------
        X : array
            The input samples.

        Returns
        -------
        X[:,self.feat2] >= self.intercept + self.slope * X[:,self.feat1]: boolean array 
            True if given point is on or above the diagonal, otherwise False.
        """
        return X[:,self.feat2] >= self.intercept + self.slope * X[:,self.feat1]

class OrthogonalSplit:
    """ The OrthogonalSplit class.

    Parameters
    ----------
    feat : int
        The index of the feature column.
    split : float
        The split threshold.

    """
    def __init__(self, feat, split):
        self.feat = feat
        self.split = split
        
    def __str__(self):
        """
        Return a string representation of the orthogonal.
        """
        return f"X[{self.feat}] >= {self.split}"
    
    def is_true(self, X):
        """
        Return whether the given points are on or to the right of the orthogonal.
        
        Parameters
        ----------
        X : array
            The input samples.

        Returns
        -------
        X[:,self.feat] >= self.split: boolean array 
            True if given point is on or to the right of the orthogonal, otherwise False.
        """
        return X[:,self.feat] >= self.split
