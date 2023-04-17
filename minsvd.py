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
A = np.array([[2,1,8],[0,2,8],[1,1,0]])
U, Sigma, Vt = my_svd(A)
print("U:\n", U)
#print("Sigma:\n", Sigma)
#print("Vt:\n", Vt)
#print("A = ",np.dot(np.dot(U,Sigma),Vt))
U_true, Sigma_true, Vt_true = np.linalg.svd(A,full_matrices=True, compute_uv=True)
print("U true:\n", U_true)
#print("Sigma true:\n", Sigma_true)
#print("Vt true:\n", Vt_true)
#print("A = ",np.dot(np.dot(U_true,np.diag(Sigma_true)),Vt_true))