import numpy as np

def my_svd(X):


    # XX_star = X@np.conjugate(X).T
    X_starX = np.conjugate(X).T@X

    eigenvalues, V = np.linalg.eig(X_starX)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    sigma = np.sqrt(eigenvalues)
    V = V[:,idx]

    #Approx U ved U ≈ X  ̃V  ̃Σ^-1

    U = X@V
    U /= sigma

    return U,sigma,np.conjugate(V).T

def MinSVDTrunc(X,k):
    # k -= 1


    # XX_star = X@np.conjugate(X).T
    X_starX = np.conjugate(X).T@X


    
    eigenvalues, V = np.linalg.eig(X_starX)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    sigma = np.sqrt(eigenvalues)
    V = V[:,idx]

    #Approx U ved U ≈ X  ̃V  ̃Σ^-1

    U = X@V
    U /= sigma

    

    return U[:,:k],sigma[:k],np.conjugate(V[:,:k]).T

def rsvd(X,k):
    
    # We sample the column space of X by generating the p-matrix
    n_c = X.shape[1]
    p = np.random.randn(n_c,k)
    Z = X@p
    
    Q, R = np.linalg.qr(Z,mode = "reduced")
    
    # Compute the SVD
    Y = Q.T @ X
    U_tilde, Sigma, VT = np.linalg.svd(Y,full_matrices=0)
    U = Q@U_tilde

    return U, Sigma, VT


""" def my_svd(A):
    # Compute the eigenvalues and eigenvectors of A^T A
    AtA = np.dot(A.T, A)
    eigenvalues, eigenvectors = np.linalg.eig(AtA)

    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # Compute the singular values
    singular_values = np.sqrt(eigenvalues)

    # Construct the Sigma matrix
    #Sigma = np.zeros(A.shape)
    #Sigma[:A.shape[1], :A.shape[1]] = np.diag(singular_values)
    Sigma = singular_values
    
    # Compute the matrix U
    U = np.dot(A, eigenvectors)
    U = U / singular_values

    # Compute the matrix V
    V = eigenvectors

    return U, Sigma, V.T """

# Example usage:
#from sklearn.datasets import load_diabetes, load_digits,load_iris
#A = np.array([[2,1,8,4,7,2,4,6],[0,2,8,5,6,9,1,3],[1,1,0,4,2,8,7,6]])
#iris = load_iris()
#X = iris.data
#U, Sigma, Vt = rsvd(X,2)
#print("U:\n", U)
#print("Sigma:\n", Sigma)
#print("Vt:\n", Vt)
#print("A = ",np.dot(np.dot(U,Sigma),Vt))
#U_true, Sigma_true, Vt_true = np.linalg.svd(X,full_matrices=0, compute_uv=True)
#print("U true:\n", U_true)
#print("Sigma true:\n", Sigma_true)
#print("Vt true:\n", Vt_true)
#print("A = ",np.dot(np.dot(U_true,np.diag(Sigma_true)),Vt_true))
#print(np.mean((U_true-U) ** 2))



