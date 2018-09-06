from sklearn import cluster
# import random
import os
import shutil
from skimage import io, transform
import numpy as np

w = 250
h = 250
c = 3


def read_img_random(path, total_count):
    file_path_list = [os.path.join(path, file_name) for file_name in os.listdir(path)
                      if os.path.isfile(os.path.join(path, file_name))]
    # print(file_path_list[0:3])
    # random.shuffle(file_path_list)
    imgs = []
    labels = []

    count = 0
    # print(file_path_list[0:3])
    while count < total_count and count < len(file_path_list):
        im = file_path_list[count]
        file_name = im.split('/')[-1]
        count += 1
        img = io.imread(im, as_grey=False)
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = transform.resize(img, (w, h))
        imgs.append(img)
        labels.append(file_name)
        if count % 1 == 0:
            print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
    print('\r', end='')
    return np.asarray(imgs, np.float32), np.asarray(labels, np.str_)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def classify_images(img_root_path, count, cat_list, img_name_list):
    for i in range(count):
        folder_path = img_root_path + str(i) + "/"
        make_dir(folder_path)
    for i in range(len(cat_list)):
        cat = cat_list[i]
        img_name = img_name_list[i]
        folder_path = img_root_path + str(cat) + "/"
        img_path = img_root_path + img_name
        shutil.copy(img_path, folder_path)


train_path = r'C:\Users\bunny\Desktop\_Training_Data\_Pictures\all/'
train_image_count = 1000
train_data, train_label = read_img_random(train_path, train_image_count)
if len(train_data.shape) == 3:
    d2_train_data = train_data.reshape(
        (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
elif len(train_data.shape) == 4:
    d2_train_data = train_data.reshape(
        (train_data.shape[0], train_data.shape[1] * train_data.shape[2] * train_data.shape[3]))
else:
    d2_train_data = []

cluster_number = 4
k_means = cluster.KMeans(n_clusters=cluster_number)
k_means.fit(d2_train_data)
# print(k_means.labels_)
# print(train_label)
for k in range(len(k_means.labels_)):
    print(str(k_means.labels_[k] + 1) + '\t' + train_label[k])
classify_images(train_path, cluster_number, k_means.labels_, train_label)
