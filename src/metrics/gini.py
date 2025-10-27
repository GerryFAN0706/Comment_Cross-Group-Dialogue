import numpy as np

def gini(x):
    """
    Compute Gini coefficient of a non-negative array.
    """
    x = np.asarray(x, dtype=float)
    x = x[x>=0]
    if x.size == 0:
        return np.nan
    if np.all(x==0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
