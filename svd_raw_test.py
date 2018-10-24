# 1. read mat & restore the first image
# 2. get COV matrix
# 3. get eigenvectors
# 4. rebuild the matrix
# 5. scatter the eigenvectors

import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from scipy.fftpack import fft, ifft

mat_path = 'mat/raw.mat'
digits = io.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]
n_samples, n_features = X.shape

w = 250
h = 250
channel = 3

X_fft = fft(X)
# X=np.absolute(X)

raw_eigenvalues, raw_eigenvectors = np.linalg.eig(np.cov(X))
fft_eigenvalues, fft_eigenvectors = np.linalg.eig(np.cov(X_fft))

raw_U, raw_s, raw_Vh = np.linalg.svd(X, full_matrices=False)
fft_U, fft_s, fft_Vh = np.linalg.svd(X_fft, full_matrices=False)

raw_s[1:] = 0
fft_s[1:] = 0

reconstructed_raw_X = np.dot(raw_U, np.dot(np.diag(raw_s), raw_Vh))
reconstructed_fft_X = np.dot(fft_U, np.dot(np.diag(fft_s), fft_Vh))
raw_image_sample = X[0].reshape(w, h, channel)
fft_image_sample = X_fft[0].real.reshape(w, h, channel)
reconstructed_raw_image_sample = reconstructed_raw_X[0].reshape(w, h, channel)
reconstructed_fft_image_sample = ifft(reconstructed_fft_X[0]).real.reshape(w, h, channel)

fig = plt.figure()
ax = fig.add_subplot(421)
ax.imshow(raw_image_sample)
ax.set_title('original_mat')

ax = fig.add_subplot(422)
ax.imshow(fft_image_sample)
ax.set_title('original_mat_fft')

ax = fig.add_subplot(423)
ax.bar(np.arange(len(raw_eigenvalues)), raw_eigenvalues)
ax.set_title('eigenvalues_raw')

ax = fig.add_subplot(424)
ax.bar(np.arange(len(fft_eigenvalues)), fft_eigenvalues)
ax.set_title('eigenvalues_fft')

ax = fig.add_subplot(425)
ax.imshow(reconstructed_raw_image_sample)
ax.set_title('reconstructed_mat')

ax = fig.add_subplot(426)
ax.imshow(reconstructed_fft_image_sample)
ax.set_title('reconstructed_mat_fft')

ax = fig.add_subplot(427)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(-0.5, 0.5)
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ev1 = raw_eigenvectors[0]
ev2 = raw_eigenvectors[1]
for i in range(ev1.shape[0]):
    ax.text(ev1[i], ev2[i], str(y[i] + 1), color=plt.cm.Set1(int(y[i])), fontdict={'size': 8})
ax.set_title('eigenvector_0,1_raw')
bottom, top = plt.ylim()  # return the current ylim

ax = fig.add_subplot(428)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(-0.5, 0.5)
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ev1 = fft_eigenvectors[0].real
ev2 = fft_eigenvectors[1].real
for i in range(ev1.shape[0]):
    ax.text(ev1[i], ev2[i], str(y[i] + 1), color=plt.cm.Set1(int(y[i])), fontdict={'size': 8})
ax.set_title('eigenvector_0,1_fft')

plt.show()

# plt.show()
