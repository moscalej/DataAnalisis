
from time import time
import matplotlib.pyplot as plt
from  matplotlib.pyplot import figure
import numpy as np
import pylab as pl
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets,neighbors,decomposition
from sklearn.decomposition import KernelPCA

n_points = 1000
X, color = datasets.make_s_curve(n_points, random_state=0)
X_lle, err = manifold.locally_linear_embedding(X, n_neighbors=12,
                                             n_components=2)
X_isomap = manifold.Isomap(n_neighbors=12, n_components=2).fit_transform(X)
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X2_PCA = pca.transform(X)

# Plot result

fig = plt.figure(figsize = (12,12))

ax = fig.add_subplot(221, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Original data")

ax = fig.add_subplot(222)
ax.scatter(X2_PCA[:, 0], X2_PCA[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data PCA')

ax = fig.add_subplot(223)
ax.scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data LLE')

ax = fig.add_subplot(224)
ax.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data ISOMAP')

plt.show()


kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
plt.title("Projection by KPCA")
plt.xlabel("1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")
fig = plt.figure(figsize = (12,12))
#plt.subplot(2, 2, 4, aspect='equal')
plt.scatter(X_back[reds, 0], X_back[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_back[blues, 0], X_back[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)

plt.show()