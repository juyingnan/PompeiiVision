from skimage import io, transform
import numpy as np
import os
import random
import feature_test
from scipy import io as sio
import k_means_test


def read_img_random(path, total_count, as_gray=False):
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
            # for angle in [0,90,180,270]:
            #     _img = transform.rotate(img, angle)
            #     _img = transform.resize(_img, (w, h))
            #     imgs.append(_img)
            #     labels.append(idx)
            img = transform.resize(img, (w, h))
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


if __name__ == '__main__':
    w = 20
    h = 20
    c = 3
    train_image_count = 10000
    train_path = r'D:\Projects\pompeii\test_20180919\svd_test/'
    train_data, train_label, train_file_paths, train_indexes = read_img_random(train_path, train_image_count,
                                                                               as_gray=True)
    np.seterr(all='ignore')

    # d2_train_data = feature_test.get_shape_index_features(train_data, size=10)
    # d2_train_data = k_means_test.get_features(train_data, whole_image_sample=True, frame_sample=False,
    #                                         global_color=True, composition=True, segment_color=True, sift=True)
    d2_train_data = k_means_test.get_raw_pixel_features(train_data)
    sio.savemat('mat/raw_' + str(w) + '.mat',
                mdict={'feature_matrix': d2_train_data,
                       'label': train_label,
                       'file_name': train_file_paths,
                       'index': train_indexes})
