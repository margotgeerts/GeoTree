import numpy as np
from matplotlib import pyplot as plt

def plot_decision_boundary(X, y, yhat, f1=0, f2=1):
    mx = np.linspace(min(X[:,f1]), max(X[:,f1]), 101)
    my = np.linspace(min(X[:,f2]), max(X[:,f2]), 101)
    xx, yy = np.meshgrid(mx, my)
    # grid = np.column_stack((xx.ravel(), yy.ravel()))
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
    plt.scatter(grid[:,f1], grid[:,f2], c=yhat(grid), alpha=.15, zorder=1)
    plt.scatter(X[:,f1], X[:, f2], c=y, s=0.5)
    return fig
 

def print_node_recurse(node, spacing):
    indent = ("|" + (" " * spacing)) * node.depth
    indent = indent[:-spacing] + "-" * spacing
    if node.is_leaf():
        print(indent, node.yhat, len(node.idx))
    else:
        print(indent, node.split, len(node.idx))
    if node.left:
        print_node_recurse(node.left)
    if node.right:
        print_node_recurse(node.right)

def print_tree(tree, spacing=3):
    print_node_recurse(tree.root)



