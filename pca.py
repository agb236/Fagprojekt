import numpy as np
from minsvd import MinSVDTrunc, my_svd
def pca(X, n_components,method,k=None):
    """
    PCA on a data matrix X using SVD

    Parameters:
    - X: Input data matrix
         numpy array of shape (n,m) where
         n = number of samples
         m = number of features

    - n_components: Number of principal components to keep (must be type int)

    - method: The kind of SVD used in the PCA
        linalg: numpy implemented SVD using numpy.linalg.svd
        svd: own implemented SVD
        trunc: own implemented truncated SVD
        rand: randomized SVD
        
    Returns:
    - X_pca: numpy array of shape (n, n_components)
        Projected data matrix.
    - eigenvectors: numpy array of shape (m, n_components)
        Eigenvectors of the data matrix.
    - explained_variance: vector of length n_components
        Variance explained by each principal component.
    """
    ## Standardizing the data
    
    # Subtract the mean value from the data
    X = X - np.mean(X, axis=0)
    
    # Divide by standard deviation
    X = X / np.std(X, axis=0)
    
    
    # Compute the SVD of the data matrix
    if method == "linalg":
        U, singular_values, V_T = np.linalg.svd(X)
    elif method == "svd":
        U, singular_values, V_T = my_svd(X)
    elif method == "trunc":
        if k is None:
            raise ValueError("Input fourth argument specifying k in the truncated SVD")
        else:
            U, singular_values, V_T = MinSVDTrunc(X,4)
    else:
        raise ValueError("Please input a valid SVD method")
    
    # Take the top n_components singular values and eigenvectors to keep
    singular_values = singular_values[:n_components]
    eigenvectors = V_T[:n_components].T

    # Project the data onto the principal components
    X_pca = np.dot(X, eigenvectors)

    # Compute the explained variance
    explained_variance = singular_values ** 2 / (X.shape[0] - 1)

    return X_pca, eigenvectors, explained_variance

