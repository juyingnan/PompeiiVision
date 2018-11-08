# 1. read mat & restore the first image
# 2. get COV matrix
# 3. get eigenvectors
# 4. rebuild the matrix
# 5. scatter the eigenvectors

import matplotlib.pyplot as plt
from scipy import io
import numpy as np

mat_path = 'mat/feature_all.mat'
digits = io.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]
# X = X.transpose()
n_samples, n_features = X.shape

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X))

U, s, Vh = np.linalg.svd(X, full_matrices=False)

s[2:] = 0

fig = plt.figure()
ax = fig.add_subplot(321)
# ax.bar(np.arange(len(X[0])), X[0])
ax.imshow(X.transpose())
if "raw" in mat_path:
    ax.set_aspect(0.1)
ax.set_title('original_mat')

ax = fig.add_subplot(322)
ax.bar(np.arange(len(eigenvalues)), eigenvalues)
ax.set_title('eigenvalues_feature')

small_edge_index = 0.2
ax = fig.add_subplot(323)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ev1 = eigenvectors[0].real
ev2 = eigenvectors[1].real
xx = X.transpose().dot(ev1)
yy = X.transpose().dot(ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)
for i in range(xx.shape[0]):
    if "raw" in mat_path:
        ax.text(xx[i], yy[i], '.', fontdict={'size': 10})
    else:
        ax.text(xx, yy, str(i), fontdict={'size': 8})
ax.set_title('features_projection')

ax = fig.add_subplot(324)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
x_mean = X.transpose().mean(axis=1, keepdims=True)
xx = (X.transpose() - x_mean).dot(ev1 - ev1.mean())
yy = (X.transpose() - x_mean).dot(ev2 - ev2.mean())
r_x, r_y = [], []
for i in range(xx.shape[0]):
    r_x.append(xx[i] / np.linalg.norm(X.transpose()[i] - x_mean[i]) / np.linalg.norm(ev1 - ev1.mean()))
    r_y.append(yy[i] / np.linalg.norm(X.transpose()[i] - x_mean[i]) / np.linalg.norm(ev2 - ev2.mean()))
edge_max = max([abs(max(r_x)), abs(min(r_x)), abs(max(r_y)), abs(min(r_y))]) * (1 + small_edge_index)
ax.set_ylim(-edge_max, edge_max)
ax.set_xlim(-edge_max, edge_max)
for i in range(xx.shape[0]):
    # ax.text(xx, yy, str(i), color=plt.cm.tab20(i), fontdict={'size': 8})
    ax.text(r_x[i], r_y[i], '.', fontdict={'size': 10})
ax.set_title('features_correlation')

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X.transpose()))
U, s, Vh = np.linalg.svd(X.transpose(), full_matrices=False)
s[2:] = 0

ax = fig.add_subplot(325)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ev1 = eigenvectors[0].real
ev2 = eigenvectors[1].real
xx = X.dot(ev1)
yy = X.dot(ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)
for i in range(X.shape[0]):
    ax.text(xx[i], yy[i], str(y[i] + 1), color=plt.cm.Set1(int(y[i])), fontdict={'size': 8})
ax.set_title('samples_projection')

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X.transpose()))
U, s, Vh = np.linalg.svd(X.transpose(), full_matrices=False)
s[2:] = 0

ax = fig.add_subplot(326)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
x_mean = X.mean(axis=1, keepdims=True)
xx = (X - x_mean).dot(ev1 - ev1.mean())
yy = (X - x_mean).dot(ev2 - ev2.mean())
r_x, r_y = [], []
for i in range(xx.shape[0]):
    r_x.append(xx[i] / np.linalg.norm(X[i] - x_mean[i]) / np.linalg.norm(ev1 - ev1.mean()))
    r_y.append(yy[i] / np.linalg.norm(X[i] - x_mean[i]) / np.linalg.norm(ev2 - ev2.mean()))
edge_max = max([abs(max(r_x)), abs(min(r_x)), abs(max(r_y)), abs(min(r_y))]) * (1 + small_edge_index)
ax.set_ylim(-edge_max, edge_max)
ax.set_xlim(-edge_max, edge_max)
for i in range(X.shape[0]):
    ax.text(r_x[i], r_y[i], str(y[i] + 1), color=plt.cm.Set1(int(y[i])), fontdict={'size': 8})
ax.set_title('samples_correlation')

plt.show()
