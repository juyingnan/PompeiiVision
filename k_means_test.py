from sklearn import cluster
# import random
import os
import shutil
from skimage import io, transform, color, exposure
import numpy as np
import csv

w = 250
h = 250
c = 3
cluster_number = 4


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


def write_csv(img_name_list, cat_list, path='csv/kmeans_{0}.csv'.format(cluster_number)):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow((["NAME", "KMEANS_CAT{0}".format(cluster_number)]))
        lines = []
        for i in range(len(img_name_list)):
            lines.append([img_name_list[i], cat_list[i]])
        writer.writerows(lines)


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


def get_raw_pixel_features(data):
    if len(data.shape) == 3:
        result = data.reshape(
            (data.shape[0], data.shape[1] * data.shape[2]))
    elif len(data.shape) == 4:
        result = data.reshape(
            (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))
    else:
        result = []
    return result


def calculate_average_hue_saturation(img):
    img_hsv = color.rgb2hsv(img)
    img_h = img_hsv[:, :, 0]
    img_s = img_hsv[:, :, 1]
    average_h = img_h.mean()
    average_s = img_s.mean()
    return [average_h, average_s]


def calculate_hue_distribution(img):
    img_hsv = color.rgb2hsv(img)
    hist, bin_centers = exposure.histogram(img_hsv, 20)
    hist_max = max(hist)
    c = 0.1
    quantized_hues_number = len([hist_value for hist_value in hist if hist_value > hist_max * c])
    # print(quantized_hues_number)
    result = list(hist)
    result.append(quantized_hues_number)
    return result


def get_color_features(data):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]
        result[-1].extend(calculate_average_hue_saturation(img))
        result[-1].extend(calculate_hue_distribution(img))
    return result


def get_composition_features(data):
    pass


def get_segment_features(data):
    pass


def get_sift_features(data):
    pass


def get_features(data, color=True, composition=True, segment=True, sift=True):
    result = []
    color_result = []
    composition_result = []
    segment_result = []
    sift_result = []
    for i in range(len(data)):
        result.append([])

    if color:
        color_result = get_color_features(data)
    if composition:
        composition_result = get_composition_features(data)
    if segment:
        segment_result = get_segment_features(data)
    if sift:
        sift_result = get_sift_features(data)

    for feature_result in [color_result, composition_result, segment_result, sift_result]:
        for i in range(len(feature_result)):
            result[i].extend(feature_result[i])
    return result


train_path = r'C:\Users\bunny\Desktop\test_20180919\unsupervised/'
train_image_count = 1000
train_data, train_label = read_img_random(train_path, train_image_count)

# d2_train_data = get_raw_pixel_features(train_data)
d2_train_data = get_features(train_data, color=True, composition=False, segment=False, sift=False)

k_means = cluster.KMeans(n_clusters=cluster_number)
k_means.fit(d2_train_data)
# print(k_means.labels_)
# print(train_label)
for k in range(len(k_means.labels_)):
    print(str(k_means.labels_[k] + 1) + '\t' + train_label[k])
classify_images(train_path, cluster_number, k_means.labels_, train_label)
write_csv(train_label, k_means.labels_ + 1)
