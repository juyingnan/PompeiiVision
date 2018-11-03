# 1. read mat & restore the first image
# 2. get COV matrix
# 3. get eigenvectors
# 4. rebuild the matrix
# 5. scatter the eigenvectors

import matplotlib.pyplot as plt
from scipy import io
import numpy as np

mat_path = 'mat/raw_20.mat'
digits = io.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]
# X = X.transpose()
n_samples, n_features = X.shape

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X))

U, s, Vh = np.linalg.svd(X, full_matrices=False)

s[2:] = 0

fig = plt.figure()
ax = fig.add_subplot(221)
# ax.bar(np.arange(len(X[0])), X[0])
ax.imshow(X.transpose())
# ax.set_aspect(0.1)
ax.set_title('original_mat')

ax = fig.add_subplot(222)
ax.bar(np.arange(len(eigenvalues)), eigenvalues)
ax.set_title('eigenvalues_feature')

small_edge = 0.1
ax = fig.add_subplot(223)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ev1 = eigenvectors[0].real
ev2 = eigenvectors[1].real
xx = X.transpose().dot(ev1)
yy = X.transpose().dot(ev2)
ax.set_ylim(min(yy) if min(yy) <= -small_edge else -small_edge, max(yy) if max(yy) >= small_edge else small_edge)
ax.set_xlim(min(xx) if min(xx) <= -small_edge else -small_edge, max(xx) if max(xx) >= small_edge else small_edge)
for i in range(xx.shape[0]):
    # ax.text(xx, yy, str(i), color=plt.cm.tab20(i), fontdict={'size': 8})
    ax.text(xx[i], yy[i], '.', fontdict={'size': 10})
ax.set_title('features')

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X.transpose()))
U, s, Vh = np.linalg.svd(X.transpose(), full_matrices=False)
s[2:] = 0

ax = fig.add_subplot(224)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ev1 = eigenvectors[0].real
ev2 = eigenvectors[1].real
xx = X.dot(ev1)
yy = X.dot(ev2)
ax.set_ylim(min(yy) if min(yy) <= -small_edge else -small_edge, max(yy) if max(yy) >= small_edge else small_edge)
ax.set_xlim(min(xx) if min(xx) <= -small_edge else -small_edge, max(xx) if max(xx) >= small_edge else small_edge)
for i in range(X.shape[0]):
    ax.text(xx[i], yy[i], str(y[i] + 1), color=plt.cm.Set1(int(y[i])), fontdict={'size': 8})
ax.set_title('samples')

plt.show()
