import os
from scipy import io as sio
import numpy as np

date = '20220303'
mat_root_dir = f'../mat/{date}'
base_mat_name = 'auto_features'
base_mat = sio.loadmat(os.path.join(mat_root_dir, base_mat_name))
base_feature = base_mat.get('feature_matrix')
print(base_feature.shape)
mat_file_names = ['manual_features']

for mat_name in mat_file_names:
    mat_path = os.path.join(mat_root_dir, mat_name)
    digits = sio.loadmat(mat_path)
    feature = digits.get('feature_matrix')
    print(feature.shape)
    base_feature = np.concatenate((base_feature, feature), axis=1)
    print(base_feature.shape)

base_mat['feature_matrix'] = base_feature
sio.savemat(file_name=os.path.join(mat_root_dir, 'auto_manual_features.mat'),
            mdict=base_mat)

