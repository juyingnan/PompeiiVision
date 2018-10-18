# reference: https://github.com/DmitryUlyanov/Multicore-TSNE

import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy import io

mat_path = 'mat/shape_index_1.mat'
digits = io.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]
n_samples, n_features = X.shape

'''t-SNE'''
tsne = TSNE(n_jobs=8, init='random', random_state=501)
X_tsne = tsne.fit_transform(X)
print("After {} iter: Org data dimension is {}. Embedded data dimension is {}".format(tsne.n_iter, X.shape[-1],
                                                                                      X_tsne.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    # plt.plot(X_norm[i, 0], X_norm[i, 1], '.', color=plt.cm.Set1(int(y[i][0])))
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(int(y[i])), fontdict={'size': 8})
plt.xticks([])
plt.yticks([])
plt.show()