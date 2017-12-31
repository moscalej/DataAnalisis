import numpy as np
import pandas as pd
np.random.seed=50
from sklearn import decomposition
from time import time
import matplotlib.pyplot as plt
from  matplotlib.pyplot import figure
import pylab as pl
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets,neighbors,decomposition
from sklearn.decomposition import KernelPCA


df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()#Return the datastructure
iris = datasets.load_iris() #this is built-in in sklearn
X=iris.data
#The class of each observation is stored in the .target attribute of the dataset.
y = iris.target


# def pca_func(X,d)
#   step 0 - Substract means
X=X.T
mean_vec = np.mean(X, axis=1)
X_zeroMean = (X.T-mean_vec).T
X.shape # should be (number of features, number of examples)

#step 1 - Calculate the empirical covariance matrix

cov_mat = X_zeroMean.dot(X_zeroMean.T) / (X.shape[1]-1)
print('Covariance matrix \n%s' %cov_mat)

#step 2 - Choose k leading eigenvectors and project
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
"""u,s,v = np.linalg.svd(X_zeroMean/np.sqrt(X.shape[1]-1))
print('Eigenvectors \n%s' %u)
ss = np.square(s)
print('\nEigenvalues \n%s' %ss)"""
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

w =np.matrix([eig_pairs[i][1] for i in range(2)]).T
print(w)
a =np.dot(X.T,w)
#step 3 - Add means
data = pd.DataFrame(X.T)
PPP=pd.DataFrame(decomposition.PCA(n_components=2).fit_transform(data))
print("123")