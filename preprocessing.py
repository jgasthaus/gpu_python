from numpy.linalg import eig
from numpy import *

def normalize(X):
    """Normalize data X (variables in rows) by subtracting the mean
    and dividing by the largest standard deviation."""
    Z = (X.T - mean(X,1)).T # subtract mean
    max_var = max(var(Z,1))
    return Z / sqrt(max_var)

def pca(X,keep=2):

    """Perfrom PCA on data X. Assumes that data points correspond to columns.
    """
    # Z = (X.T - mean(X,1)).T # subtract mean
    C = dot(X,X.T)
    V,D = eig(C)
    B = D[:,0:keep]
    return dot(B.T,X)

