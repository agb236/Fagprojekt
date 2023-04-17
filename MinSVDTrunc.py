import numpy as np

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

# X = np.array([[2,3,4],[1,5,1],[8,9,0]])
# X = np.array([[2,1,8],[0,2,8],[1,1,0]])

# U,sigma,Vh = MinSVDTrunc(X,k=2)
# U2,sigma2,Vh2 = np.linalg.svd(X)

# print(U@np.diag(sigma)@Vh)
# print('-----------------------------------------')
# print(U2)