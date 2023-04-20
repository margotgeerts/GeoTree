import numpy as np
from matplotlib import pyplot as plt

def plot_decision_boundary(X, y, model, f1=0, f2=1):
    """
    Plot the decision boundaries on a grid using two features.

    Parameters
    ----------
    X : np.array, shape (n_samples, n_features)
        A numpy array containing the input samples. 
    y : np.array, shape (n_samples,)
        The target variable.
    model : object
        A trained model that has a 'predict' function.
    n : int
        The total number of samples in the dataset.
    f1 : int
        The index of the first feature column.
    f2 : int
        The index of the second feature column.
    
    Returns
    -------
    fig : matplotlib.pyplot.figure object
    """
    mx = np.linspace(min(X[:,f1]), max(X[:,f1]), 101)
    my = np.linspace(min(X[:,f2]), max(X[:,f2]), 101)
    xx, yy = np.meshgrid(mx, my)

    grid = np.zeros((10201,X.shape[1]))

    # fill grid with averages of other columns
    for i in range(0, X.shape[1]):
        # first define new column
        new_col = None
        if i == f1:
            new_col = xx.ravel()
        elif i == f2:
            new_col = yy.ravel()
        else:
            new_col = np.array([np.mean(X[:,i]) for _ in range(10201)])
        
        # then add new column to grid
        grid[:,i] = new_col
    fig = plt.figure()
    plt.scatter(grid[:,f1], grid[:,f2], c=model.predict(grid), alpha=.15, zorder=1)
    plt.scatter(X[:,f1], X[:, f2], c=y, s=0.5)
    return fig
 



