from skimage import io, transform
import numpy as np
import os
import random
import feature_test
from scipy import io as sio
import k_means_test


def read_img_random(path, total_count):
    cate = [path + folder for folder in os.listdir(path) if os.path.isdir(path + folder)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        count = 0
        file_path_list = [os.path.join(folder, file_name) for file_name in os.listdir(folder)
                          if os.path.isfile(os.path.join(folder, file_name))]
        # print(file_path_list[0:3])
        random.shuffle(file_path_list)
        # print(file_path_list[0:3])
        while count < total_count and count < len(file_path_list):
            im = file_path_list[count]
            count += 1
            img = io.imread(im)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            # for angle in [0,90,180,270]:
            #     _img = transform.rotate(img, angle)
            #     _img = transform.resize(_img, (w, h))
            #     imgs.append(_img)
            #     labels.append(idx)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
            if count % 100 == 0:
                print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
        print('\r', end='')
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


if __name__ == '__main__':
    w = 250
    h = 250
    c = 3
    train_image_count = 1000
    category_count = 4
    train_path = r'C:\Users\bunny\Desktop\test_20180919\feature_mat\TRAIN/'
    train_data, train_label = read_img_random(train_path, train_image_count)
    np.seterr(all='ignore')

    # d2_train_data = feature_test.get_shape_index_features(train_data, size=2)
    # d2_train_data = k_means_test.get_features(train_data, whole_image_sample=True, frame_sample=False, global_color=False,
    #                            composition=False, segment_color=False, sift=True)
    d2_train_data = k_means_test.get_raw_pixel_features(train_data)
    sio.savemat('mat/raw.mat',
                mdict={'feature_matrix': d2_train_data, 'label': train_label})
