import os

import numpy as np
from scipy import io as sio
from skimage import io, transform

import feature_test
from classification import k_means_test


def read_img_random(path, total_count, as_gray=False, resize=None):
    cate = [path + folder for folder in os.listdir(path) if os.path.isdir(path + folder)]
    imgs = []
    labels = []
    file_names = []
    indexes = []
    roman_label = ['I', 'II', 'III', 'IV']
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        count = 0
        file_path_list = [os.path.join(folder, file_name) for file_name in os.listdir(folder)
                          if os.path.isfile(os.path.join(folder, file_name))]
        # print(file_path_list[0:3])
        # random.shuffle(file_path_list)
        # print(file_path_list[0:3])
        while count < total_count and count < len(file_path_list):
            im = file_path_list[count]
            count += 1
            img = io.imread(im, as_gray=as_gray)
            if resize is not None:
                img = transform.resize(img, resize, anti_aliasing=True)
            imgs.append(img)
            labels.append(idx)
            file_names.append(im.split('\\')[-1])
            indexes.append(count)
            print(roman_label[idx] + '-' + str(count), file_names[-1])
            if count % 100 == 0:
                print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
        print('\r', end='')
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32), np.asarray(file_names, np.str_), np.asarray(
        indexes, np.int32)


def save_raw_mat(raw_data, label, file_names, indexes, file_name):
    d2_train_data = k_means_test.get_raw_pixel_features(raw_data)
    print('raw mat size: ', d2_train_data.shape)
    sio.savemat(file_name,
                mdict={'feature_matrix': d2_train_data,
                       'label': label,
                       'file_name': file_names,
                       'index': indexes})


def save_shape_index_mat(raw_data, label, file_names, indexes, size, file_name):
    d2_train_data = feature_test.get_shape_index_features(raw_data, size=size)
    print('shape index mat size: ', d2_train_data.shape)
    sio.savemat(file_name,
                mdict={'feature_matrix': d2_train_data,
                       'label': label,
                       'file_name': file_names,
                       'index': indexes})


def save_key_point_mat(raw_data, label, file_names, indexes, file_name):
    d2_train_data = k_means_test.get_features(raw_data, whole_image_sample=True, frame_sample=False, global_color=False,
                                              composition=False, segment_color=False, sift=True)
    print('key point mat size: ', d2_train_data.shape)
    sio.savemat(file_name,
                mdict={'feature_matrix': d2_train_data,
                       'label': label,
                       'file_name': file_names,
                       'index': indexes})


def save_all_feature_mat(raw_data, label, file_names, indexes, file_name):
    d2_train_data = k_means_test.get_features(raw_data, whole_image_sample=True, frame_sample=False, global_color=True,
                                              composition=True, segment_color=True, sift=True)
    print('all feature mat size: ', d2_train_data.shape)
    sio.savemat(file_name,
                mdict={'feature_matrix': d2_train_data,
                       'label': label,
                       'file_name': file_names,
                       'index': indexes})


if __name__ == '__main__':
    w = 500
    h = w
    c = 3
    train_image_count = 10000
    train_path = r'D:\Projects\pompeii\20190405\svd_500_submean/'
    for is_gray in [True, False]:
        train_data, train_label, train_file_paths, train_indexes = read_img_random(train_path, train_image_count,
                                                                                   as_gray=is_gray, resize=(h, w))
        np.seterr(all='ignore')

        save_shape_index_mat(train_data, train_label, train_file_paths, train_indexes, size=10,
                             file_name='mat/shape_index_' + str(10) + ('_g' if is_gray else '') + '.mat')
        save_key_point_mat(train_data, train_label, train_file_paths, train_indexes,
                           file_name='mat/key_point_' + str(12) + ('_g' if is_gray else '') + '.mat')
        if not is_gray:
            save_all_feature_mat(train_data, train_label, train_file_paths, train_indexes,
                                 file_name='mat/all_feature.mat')

        for edge in [20, 50, 100, 250]:
            w = h = edge
            train_data, train_label, train_file_paths, train_indexes = read_img_random(train_path, train_image_count,
                                                                                       as_gray=is_gray, resize=(h, w))

            save_raw_mat(train_data, train_label, train_file_paths, train_indexes,
                         'mat/raw_' + str(w) + ('_g' if is_gray else '') + '.mat')
