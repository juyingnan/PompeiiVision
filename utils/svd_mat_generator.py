import os
import csv

import numpy as np
from scipy import io as sio
from skimage import io, transform
from skimage.color import gray2rgb

import feature_test
from classification import k_means_test


def read_csv(path):
    # init result
    manual_features_dict = {}

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        # skip header
        num_cols = len(next(reader))

        # assign file into sets
        for row in reader:
            filename = row[0]
            if len(filename) == 0:
                break
            location = row[1]
            manual_features = []
            for i in range(2, num_cols):
                if len(row[i]) > 0:
                    manual_features.append(int(row[i]))
                else:
                    manual_features.append(-1)
            if filename not in manual_features_dict:
                manual_features_dict[filename] = {
                    "location": location,
                    "manual_features": manual_features
                }
    return manual_features_dict
    # return np.asarray(file_names, np.str_), np.asarray(styles, np.int8), np.asarray(manual_features, np.int8)


def read_img_random(path, file_names, as_gray=False, resize=None):
    imgs = list()
    # roman_label = ['I', 'II', 'III', 'IV']
    print('reading the images:%s' % path)
    for file_name in file_names:
        file_path = file_name
        img = io.imread(file_path, as_gray=as_gray)
        if resize is not None:
            img = transform.resize(img, resize, anti_aliasing=False)
        # io.imsave(file_path, img)
        if img.shape[-1] != 3 and not as_gray:
            print(file_path)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            else:
                img = gray2rgb(img)
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

    # csv_file_path = r'C:\Users\bunny\Desktop\Database_Revised.txt'
    # file_name_list, style_list, manual_features_list = read_csv(csv_file_path)

    file_name_list = []
    img_full_path_list = []
    relative_path_list = []
    style_list = []
    label_list = []
    location_list = []
    manual_features_list = []
    i = -1

    image_root = r'C:\Users\bunny\Desktop\_Reorganized_Data'
    csv_path = r'C:\Users\bunny\Desktop\manual.csv'
    manual_features_all = read_csv(csv_path)
    for (dirpath, dirnames, filenames) in os.walk(image_root):
        img_full_path_list += [os.path.join(dirpath, file) for file in filenames
                               if (file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')))]
        file_name_list += [file for file in filenames
                           if (file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')))]
        relative_path_list += [f'{dirpath.split(os.path.sep)[-1]}/{file}' for file in filenames
                               if (file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')))]
        style_list += [dirpath.split(os.path.sep)[-1] for file in filenames
                       if (file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')))]
        label_list += [i for file in filenames
                       if (file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')))]
        i += 1

    index_list = [list(style_list)[:i + 1].count(style_list[i]) for i in range(len(style_list))]

    # manual features
    for filename in file_name_list:
        filename_prefix = os.path.splitext(filename)[0]
        if filename_prefix in manual_features_all:
            manual_features_list.append(manual_features_all[filename_prefix]["manual_features"])
            location_list.append(manual_features_all[filename_prefix]["location"])
        else:
            print(filename)

    for is_gray in [False]:
        raw_pixel_list = read_img_random(image_root, img_full_path_list, as_gray=is_gray, resize=(h, w))

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

        save_to_dir = '../mat/20210308/'

        sio.savemat(file_name=os.path.join(save_to_dir, f'auto_features{"_g" if is_gray else ""}.mat'),
                    mdict={'feature_matrix': np.concatenate((shape_index_feature_list, key_point_feature_list), axis=1),
                           'label': label_list,
                           'style': style_list,
                           'location': location_list,
                           'file_name': file_name_list,
                           'relative_file_name': relative_path_list,
                           'index': index_list})
        sio.savemat(file_name=os.path.join(save_to_dir, f'manual_features{"_g" if is_gray else ""}.mat'),
                    mdict={'feature_matrix': manual_features_list,
                           'label': label_list,
                           'style': style_list,
                           'location': location_list,
                           'file_name': file_name_list,
                           'relative_file_name': relative_path_list,
                           'index': index_list})
        # sio.savemat(file_name=os.path.join(save_to_dir, f'auto+manual_features{"_g" if is_gray else ""}.mat'),
        #             mdict={'feature_matrix': np.concatenate(
        #                 (shape_index_feature_list, key_point_feature_list, manual_features_list), axis=1),
        # 'label': label_list,
        # 'style': style_list,
        # 'file_name': file_name_list,
        # 'relative_file_name': relative_path_list,
        # 'index': index_list})

        for edge in [20, 50]:
            w = h = edge
            raw_pixel_list = read_img_random(image_root, img_full_path_list, as_gray=is_gray, resize=(h, w))
            d2_raw_pixel_list = k_means_test.get_raw_pixel_features(raw_pixel_list)
            print('raw mat size: ', d2_raw_pixel_list.shape)
            sio.savemat(file_name=os.path.join(save_to_dir, f'raw_{w}{"_g" if is_gray else ""}.mat'),
                        mdict={'feature_matrix': d2_raw_pixel_list,
                               'label': label_list,
                               'style': style_list,
                               'location': location_list,
                               'file_name': file_name_list,
                               'relative_file_name': relative_path_list,
                               'index': index_list})
            # sio.savemat(file_name=os.path.join(save_to_dir, f'raw+manual_features_{w}{"_g" if is_gray else ""}.mat'),
            #             mdict={'feature_matrix': np.concatenate((d2_raw_pixel_list, manual_features_list), axis=1),
            # 'label': label_list,
            # 'style': style_list,
            # 'file_name': file_name_list,
            # 'relative_file_name': relative_path_list,
            # 'index': index_list})

            #
            # save_raw_mat(raw_pixel_list, train_label, train_file_paths, train_indexes,
            #              '../mat/raw_' + str(w) + ('_g' if is_gray else '') + '.mat')
