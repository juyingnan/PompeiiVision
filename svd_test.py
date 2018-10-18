# 1. read mat & restore the first image
# 2. get COV matrix
# 3. get eigenvectors
# 4. rebuild the matrix
# 5. scatter the eigenvectors

import matplotlib.pyplot as plt
from scipy import io
import numpy as np
import k_means_test

mat_path = 'mat/raw_250.mat'
digits = io.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]
n_samples, n_features = X.shape

w = 250
h = 250
channel = 3

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X))
plt.bar(np.arange(len(eigenvalues)), eigenvalues)
plt.show()

U, s, Vh = np.linalg.svd(X, full_matrices=False)
# assert np.allclose(X, np.dot(U, np.dot(np.diag(s), Vh)))

s[2:] = 0
reconstructed_X = np.dot(U, np.dot(np.diag(s), Vh))
image_0 = X[0].reshape(w, h, channel)
reconstructed_image_0 = reconstructed_X[0].reshape(w, h, channel)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(image_0)
ax.set_title('original_mat')

ax = fig.add_subplot(122)
ax.imshow(reconstructed_image_0)
ax.set_title('reconstructed_mat')
plt.show()

normalized_ev1 = k_means_test.normalize_features(eigenvectors[0], v_min=0.0, v_max=1.0)
normalized_ev2 = k_means_test.normalize_features(eigenvectors[1], v_min=0.0, v_max=1.0)
for i in range(normalized_ev1.shape[0]):
    # plt.plot(X_norm[i, 0], X_norm[i, 1], '.', color=plt.cm.Set1(int(y[i][0])))
    plt.text(normalized_ev1[i], normalized_ev2[i], str(y[i] + 1), color=plt.cm.Set1(int(y[i])), fontdict={'size': 8})
plt.show()
