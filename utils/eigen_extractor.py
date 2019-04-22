import numpy as np
from scipy import io


def eigen_discard(matrix, discard_list, need_reshape=False):
    if need_reshape:
        _u, _s, _vh = np.linalg.svd(matrix.reshape(matrix.shape[0], -1), full_matrices=False)
    else:
        _u, _s, _vh = np.linalg.svd(matrix, full_matrices=False)
    for index in discard_list:
        _s[index] = 0
    result = np.dot(_u * _s, _vh)
    return result


if __name__ == '__main__':
    root_path = '../mat/'
    file_name = 'raw_50.mat'
    mat_path = root_path + file_name
    digits = io.loadmat(mat_path)
    X = digits.get('feature_matrix')
    X = X.reshape(X.shape[0], -1)
    n_samples, n_features = X.shape
    print("{} samples, {} features".format(n_samples, n_features))

    discard_eigen_list = [0, 1, 2]
    reconstruct_mat = eigen_discard(X, discard_eigen_list, need_reshape=False)
    print(reconstruct_mat.shape)
    digits['feature_matrix'] = reconstruct_mat
    io.savemat(mat_path.replace('.mat', '-{}.mat'.format('-'.join([str(item) for item in discard_eigen_list]))),
               mdict=digits)
