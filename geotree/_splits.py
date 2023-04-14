import numpy as np

def euclidean(p1x, p1y, p2x, p2y):
    return ((p1x - p2x)**2 + (p1y - p2y)**2)**(.5)

class EllipseSplit:
    def __init__(self, feat1, feat2, focal1, focal2, dist, bbox, regf=0):
        self.feat1 = feat1
        self.feat2 = feat2
        self.focal1 = np.copy(focal1).astype(np.float32)
        self.focal2 = np.copy(focal2).astype(np.float32)
        self.dist = dist
        self.regf = regf
        self.bbox = bbox

    def __str__(self):
        return f"d(X[{self.feat1}], {self.focal1}) + d(X[{self.feat2}], {self.focal2}) >= {self.dist}"

    def ellipse_area(self):
        b = self.dist
        distf1f2 = euclidean(self.focal1[0],self.focal1[1],self.focal2[0],self.focal2[1])
        a = np.sqrt((b**2) + ((distf1f2/2)**2))
        return (np.pi * a * b) / self.bbox
    
    def is_true(self, X):
        dist_1 = np.sqrt((X[:, self.feat1] - self.focal1[0])**2 + (X[:, self.feat2] - self.focal1[1])**2)
        dist_2 = np.sqrt((X[:, self.feat1] - self.focal2[0])**2 + (X[:, self.feat2] - self.focal2[1])**2)
        return dist_1 + dist_2 >= self.dist
    
    def evaluate_ellipse(self, gain, n_t, n):
        if self.regf > 0:
          regularisation = ((n_t/n) * self.regf * self.ellipse_area())
          if not gain or regularisation >= gain:
              return 0.0
          return (gain - regularisation)
        
        return gain
          

class DiagonalSplit:
    def __init__(self, feat1, feat2, intercept, slope):
        self.feat1 = feat1
        self.feat2 = feat2
        self.intercept = intercept
        self.slope = slope

    def __str__(self):
        return f"X[{self.feat2}] >= {self.intercept} + {self.slope} * X[{self.feat1}]"

    def is_true(self, X):
        return X[:,self.feat2] >= self.intercept + self.slope * X[:,self.feat1]

class OrthogonalSplit:
    def __init__(self, feat, split):
        self.feat = feat
        self.split = split
        
    def __str__(self):
        return f"X[{self.feat}] >= {self.split}"
    
    def is_true(self, X):
        return X[:,self.feat] >= self.split
