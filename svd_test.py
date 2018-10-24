# 1. read mat & restore the first image
# 2. get COV matrix
# 3. get eigenvectors
# 4. rebuild the matrix
# 5. scatter the eigenvectors

import matplotlib.pyplot as plt
from scipy import io
import numpy as np

mat_path = 'mat/shape_index_feature_10.mat'
digits = io.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]
n_samples, n_features = X.shape

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X))

U, s, Vh = np.linalg.svd(X, full_matrices=False)

s[2:] = 0

fig = plt.figure()
ax = fig.add_subplot(221)
ax.bar(np.arange(len(X[0])), X[0])
ax.set_title('original_mat')

ax = fig.add_subplot(222)
ax.bar(np.arange(len(eigenvalues)), eigenvalues)
ax.set_title('eigenvalues_feature')

ax = fig.add_subplot(224)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(-0.5, 0.5)
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ev1 = eigenvectors[0]
ev2 = eigenvectors[1]
for i in range(ev1.shape[0]):
    ax.text(ev1[i], ev2[i], str(y[i] + 1), color=plt.cm.Set1(int(y[i])), fontdict={'size': 8})
ax.set_title('eigenvector_0,1_feature')
bottom, top = plt.ylim()

plt.show()
