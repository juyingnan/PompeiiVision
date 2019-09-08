import os
import csv

import numpy as np
from scipy import io as sio
from skimage import io, transform

import feature_test
from classification import k_means_test


def read_csv(path):
    # init result
    file_names = list()
    styles = list()
    manual_features = list()

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        # skip header
        num_cols = len(next(reader))

        # assign file into sets
        for row in reader:
            filename = row[0]
            if len(filename) == 0:
                break
            file_names.append(filename)
            style = int(row[1])
            styles.append(style)
            manual_features.append(list())
            for i in range(2, num_cols):
                manual_features[-1].append(int(row[i]))

    return np.asarray(file_names, np.str_), np.asarray(styles, np.int8), np.asarray(manual_features, np.int8)


def read_img_random(path, file_names, as_gray=False, resize=None):
    imgs = list()
    # roman_label = ['I', 'II', 'III', 'IV']
    print('reading the images:%s' % path)
    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        img = io.imread(file_path, as_gray=as_gray)
        if resize is not None:
            img = transform.resize(img, resize, anti_aliasing=True)
        # io.imsave(file_path, img)
        if img.shape[-1] != 3 and not as_gray:
            print(file_path)
        imgs.append(img)

    return np.asarray(imgs, np.float32)


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

    csv_file_path = r'C:\Users\bunny\Desktop\Database_Revised.txt'
    file_name_list, style_list, manual_features_list = read_csv(csv_file_path)
    index_list = [list(style_list)[:i + 1].count(style_list[i]) for i in range(len(style_list))]

    image_root = r'C:\Users\bunny\Desktop\svd_test_raw/'
    for is_gray in [False]:
        raw_pixel_list = read_img_random(image_root, file_name_list, as_gray=is_gray, resize=(h, w))

        np.seterr(all='ignore')

        # save_shape_index_mat(raw_pixel_list, train_label, train_file_paths, train_indexes, size=10,
        #                      file_name='../mat/shape_index_' + str(10) + ('_g' if is_gray else '') + '.mat')
        # save_key_point_mat(raw_pixel_list, train_label, train_file_paths, train_indexes,
        #                    file_name='../mat/key_point_' + str(12) + ('_g' if is_gray else '') + '.mat')
        # if not is_gray:
        #     save_all_feature_mat(raw_pixel_list, train_label, train_file_paths, train_indexes,
        #                          file_name='../mat/all_feature.mat')

        shape_index_feature_list = feature_test.get_shape_index_features(raw_pixel_list, size=10)
        print('shape index mat size: ', shape_index_feature_list.shape)
        key_point_feature_list = k_means_test.get_features(raw_pixel_list,
                                                           whole_image_sample=True, frame_sample=False,
                                                           global_color=False, composition=False, segment_color=False,
                                                           sift=True)
        print('key point mat size: ', key_point_feature_list.shape)

        sio.savemat(file_name='../mat/auto_features{}.mat'.format('_g' if is_gray else ''),
                    mdict={'feature_matrix': np.concatenate((shape_index_feature_list, key_point_feature_list), axis=1),
                           'label': style_list,
                           'file_name': file_name_list,
                           'index': index_list})
        sio.savemat(file_name='../mat/manual_features{}.mat'.format('_g' if is_gray else ''),
                    mdict={'feature_matrix': manual_features_list,
                           'label': style_list,
                           'file_name': file_name_list,
                           'index': index_list})
        sio.savemat(file_name='../mat/auto+manual_features{}.mat'.format('_g' if is_gray else ''),
                    mdict={'feature_matrix': np.concatenate(
                        (shape_index_feature_list, key_point_feature_list, manual_features_list), axis=1),
                        'label': style_list,
                        'file_name': file_name_list,
                        'index': index_list})

        for edge in [20, 50]:
            w = h = edge
            raw_pixel_list = read_img_random(image_root, file_name_list, as_gray=is_gray, resize=(h, w))
            d2_raw_pixel_list = k_means_test.get_raw_pixel_features(raw_pixel_list)
            print('raw mat size: ', d2_raw_pixel_list.shape)
            sio.savemat(file_name='../mat/raw_{}{}.mat'.format(w, '_g' if is_gray else ''),
                        mdict={'feature_matrix': d2_raw_pixel_list,
                               'label': style_list,
                               'file_name': file_name_list,
                               'index': index_list})
            sio.savemat(file_name='../mat/raw_{}+manual_features{}.mat'.format(w, '_g' if is_gray else ''),
                        mdict={'feature_matrix': np.concatenate((d2_raw_pixel_list, manual_features_list), axis=1),
                               'label': style_list,
                               'file_name': file_name_list,
                               'index': index_list})

            #
            # save_raw_mat(raw_pixel_list, train_label, train_file_paths, train_indexes,
            #              '../mat/raw_' + str(w) + ('_g' if is_gray else '') + '.mat')
