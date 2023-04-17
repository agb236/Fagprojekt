import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.datasets import load_diabetes, load_digits,load_iris
import xlrd
from mpl_toolkits.mplot3d import Axes3D
from minsvd import MinSVDTrunc, my_svd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
from scipy.io import loadmat
from pca import pca


print("___________________________________")
# Load the iris dataset
iris = load_iris()
diabetes = load_diabetes()
digits =  load_digits()

X = iris.data
X_pca, eigenvectors, explained_variance = pca(X, 2, "svd")



#
def plot_pca(data,n_comp,meth):
    X = data.data
    y = data.target

    attributeNames = data.feature_names
    classLabels = data.target_names
    classNames = sorted(set(classLabels))
    classDict = dict(zip(classNames, range(3)))
    
    N = len(y)
    M = len(attributeNames)
    C = len(classNames)
    print("Number of obs.:",N,"Attribute names: ",M,"Class names: ",C)
    
    X_pca, eigenvectors, explained_variance = pca(X, n_components=n_comp,method=meth)
    
    if n_comp == 2:
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
        ax.set_xlabel('PC1 ({}%)'.format(round(100 * explained_variance[0] / np.sum(explained_variance), 2)))
        ax.set_ylabel('PC2 ({}%)'.format(round(100 * explained_variance[1] / np.sum(explained_variance), 2)))
        ax.set_title('PCA on Iris Dataset')
        plt.show()
    elif n_comp == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y)
        ax.set_xlabel('PC1 ({}%)'.format(round(100 * explained_variance[0] / np.sum(explained_variance), 2)))
        ax.set_ylabel('PC2 ({}%)'.format(round(100 * explained_variance[1] / np.sum(explained_variance), 2)))
        ax.set_zlabel('PC3 ({}%)'.format(round(100 * explained_variance[2] / np.sum(explained_variance), 2)))
        ax.set_title('PCA on Iris Dataset')
        plt.show()
    else:
        print("No more than 3 or less than 2 principal components can be plotted")
    
    print("Succesfull plotting")

#plot_pca(digits,3,svd)






#___________________________________________________________________________________
"""mat_data = loadmat('/Users/agb/Desktop/Machine_Learning/02450Toolbox_Python/Data/wine2.mat')
X = mat_data['X']
y = mat_data['y']

attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0] for name in mat_data['classNames'].squeeze()]
N, M = X.shape
C = len(classNames)
print(N,M,C)
X_pca, eigenvectors, explained_variance, S = pca(X, n_components=11)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y)
ax.set_xlabel('PC1 ({}%)'.format(round(100 * explained_variance[0] / np.sum(explained_variance), 2)))
ax.set_ylabel('PC2 ({}%)'.format(round(100 * explained_variance[1] / np.sum(explained_variance), 2)))
ax.set_zlabel('PC3 ({}%)'.format(round(100 * explained_variance[2] / np.sum(explained_variance), 2)))
ax.set_title('PCA')
plt.show()

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
ax.set_xlabel('PC1 ({}%)'.format(round(100 * explained_variance[0] / np.sum(explained_variance), 2)))
ax.set_ylabel('PC2 ({}%)'.format(round(100 * explained_variance[1] / np.sum(explained_variance), 2)))
ax.set_title('PCA on Iris Dataset')
plt.show()

rho = (S*S) / (S*S).sum() 
threshold = 0.9

plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-') # len(rho)+1 fordi vi 0 indekserer
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
    plt.show()"""


#________________________________________________________













""" 
doc = xlrd.open_workbook('/Users/agb/Desktop/Machine_Learning/02450Toolbox_Python/Data/nanonose.xls').sheet_by_index(0)
# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(0, 3, 11)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(0, 2, 92)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(5))) # Tildeler tal til hver klasse
 
# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])
# Dvs. her finder den det tilsvarende tal til den class der er angivet (vand, ethanol osv.)
# Det er altså det de mener når de laver klassificeringsdata om til tal

# Preallocate memory, then extract excel data to matrix X
X = np.empty((90, 8))
for i, col_id in enumerate(range(3, 11)):
    X[:, i] = np.asarray(doc.col_values(col_id, 2, 92))

print(X.shape) # Checker om X har de rigtige dimensioner

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)
print(N,M,C)

# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
j = 2
i = 0
f = figure()
title('NanoNose data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()

X_pca, eigenvectors, explained_variance = pca(X, n_components=3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y)
ax.set_xlabel('PC1 ({}%)'.format(round(100 * explained_variance[0] / np.sum(explained_variance), 2)))
ax.set_ylabel('PC2 ({}%)'.format(round(100 * explained_variance[1] / np.sum(explained_variance), 2)))
ax.set_zlabel('PC3 ({}%)'.format(round(100 * explained_variance[2] / np.sum(explained_variance), 2)))
ax.set_title('PCA')
plt.show()"""











""" X = iris.data
y = iris.target

attributeNames = iris.feature_names

classLabels = iris.target_names
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(3)))

N = len(y)
M = len(attributeNames)
C = len(classNames)
print(N,M,C)

# Apply PCA with n principal components
X_pca, eigenvectors, explained_variance = pca(X, n_components=2)

# Visualize the projected data using scatter plots
fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
ax.set_xlabel('PC1 ({}%)'.format(round(100 * explained_variance[0] / np.sum(explained_variance), 2)))
ax.set_ylabel('PC2 ({}%)'.format(round(100 * explained_variance[1] / np.sum(explained_variance), 2)))
ax.set_title('PCA on Iris Dataset')
plt.show() """

""" fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y)
ax.set_xlabel('PC1 ({}%)'.format(round(100 * explained_variance[0] / np.sum(explained_variance), 2)))
ax.set_ylabel('PC2 ({}%)'.format(round(100 * explained_variance[1] / np.sum(explained_variance), 2)))
ax.set_zlabel('PC3 ({}%)'.format(round(100 * explained_variance[2] / np.sum(explained_variance), 2)))
ax.set_title('PCA on Iris Dataset')
plt.show() """
