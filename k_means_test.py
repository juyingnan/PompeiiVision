from sklearn import cluster
# import random
import os
import shutil
from skimage import io, transform, color, exposure, segmentation, feature
import numpy as np
import csv

w = 250
h = 250
c = 3
cluster_number = 4
color_bin_count = 10
composition_feature_count = 3
sift_feature_count = 10


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
    hist, bin_centers = exposure.histogram(img_hsv, color_bin_count)
    hist_max = max(hist)
    _c = 0.1
    quantized_hues_number = len([hist_value for hist_value in hist if hist_value > hist_max * _c])
    # print(quantized_hues_number)
    result = list(hist)
    result.append(quantized_hues_number)
    return result


def get_cropped_images(img):
    upper_index = 0.35
    lower_index = 0.2
    lr_index = 0.2
    _h, _w, _c = img.shape
    upper_bound = int(_h * upper_index)
    lower_bound = int(_h * (1 - lower_index))
    left_bound = int(_w * lr_index)
    right_bound = int(_w * (1 - lr_index))

    # get up img
    up_img = img[:upper_bound, :]
    # get down img
    down_img = img[lower_bound:, :]
    # get left img
    left_img = img[upper_bound:lower_bound, :left_bound]
    # get right img
    right_img = img[upper_bound:lower_bound, right_bound:]
    # get central img
    central_img = img[upper_bound:lower_bound, left_bound:right_bound]

    return [up_img, down_img, left_img, right_img, central_img]


def calculate_n_max_seg_value(segments, n):
    flatten_segment = segments.flatten()
    bin_count_list = list(np.bincount(flatten_segment))
    for i in range(len(bin_count_list)):
        bin_count_list[i] = (i, bin_count_list[i])
    sorted_bin_count_list = sorted(bin_count_list, reverse=True, key=lambda count: count[1])
    return sorted_bin_count_list[:n]


def calculate_seg_mass_center(segments, max_seg_value_list):
    result = []
    for seg_value_pair in max_seg_value_list:
        segment_value, segment_value_count = seg_value_pair
        seg_x = 0
        seg_y = 0
        for i in range(len(segments)):
            row = segments[i]
            for j in range(len(row)):
                if row[j] == segment_value:
                    seg_x += j
                    seg_y += i
        seg_x /= segment_value_count
        seg_y /= segment_value_count

        seg_variance = 0
        seg_skewness = 0
        for i in range(len(segments)):
            row = segments[i]
            for j in range(len(row)):
                if row[j] == segment_value:
                    seg_variance += (j - seg_x) ** 2 + (i - seg_y) ** 2
                    seg_skewness += (j - seg_x) ** 3 + (i - seg_y) ** 3
        seg_variance /= segment_value_count
        seg_skewness /= segment_value_count
        result.extend([seg_x, seg_y, seg_variance, seg_skewness])
    return result


def calculate_segmentation_mass_center(img):
    segments_fz = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    max_seg_value_list = calculate_n_max_seg_value(segments_fz, composition_feature_count)
    max_seg_mass_center_list = calculate_seg_mass_center(segments_fz, max_seg_value_list)
    # import matplotlib.pyplot as plt
    # plt.imshow(segments_fz)
    # plt.show()
    return max_seg_mass_center_list


def calculate_sift(img, n):
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = color.rgb2gray(img)
    detector = feature.CENSURE()
    detector.detect(img)
    key_point_list = []
    for i in range(len(detector.scales)):
        key_point_list.append((detector.keypoints[i], detector.scales[i]))
    sorted_key_point_list = sorted(key_point_list, reverse=True, key=lambda count: count[1])
    # result = [position for key_point in sorted_key_point_list[:n] for position in key_point[0]]
    result = [key_point[1] for key_point in sorted_key_point_list[:n]]
    while len(result) < n * 1:
        result.append(0)
    return result


def get_color_features(data):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]
        result[-1].extend(calculate_average_hue_saturation(img))
        result[-1].extend(calculate_hue_distribution(img))

        # left, right, up, down
        cropped_imgs = get_cropped_images(img)
        for cropped_img in cropped_imgs:
            result[-1].extend(calculate_average_hue_saturation(cropped_img))
            result[-1].extend(calculate_hue_distribution(cropped_img))

    return result


def get_composition_features(data):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]
        result[-1].extend(calculate_segmentation_mass_center(img))

        # cropped_imgs = get_cropped_images(img)
        # for cropped_img in cropped_imgs:
        #     result[-1].extend(calculate_segmentation_mass_center(cropped_img))

    return result


def get_segment_features(data):
    pass


def get_sift_features(data):
    result = []
    for i in range(len(data)):
        img = data[i]
        result.append(calculate_sift(img, sift_feature_count))
    return result


def normalize_features(data, v_max=1.0, v_min=0.0):
    data_array = np.asarray(data, np.float32)
    mins = 0  # np.min(data_array, axis=0)
    maxs = np.max(data_array, axis=0)
    rng = maxs - mins
    result = v_max - ((v_max - v_min) * (maxs - data_array) / rng)
    return result


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

    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


train_path = r'C:\Users\bunny\Desktop\test_20180919\unsupervised/'
train_image_count = 1000
train_data, train_label = read_img_random(train_path, train_image_count)

# d2_train_data = get_raw_pixel_features(train_data)
d2_train_data = get_features(train_data, color=True, composition=True, segment=False, sift=True)

k_means = cluster.KMeans(n_clusters=cluster_number)
k_means.fit(d2_train_data)
# print(k_means.labels_)
# print(train_label)
for k in range(len(k_means.labels_)):
    print(str(k_means.labels_[k] + 1) + '\t' + train_label[k])
classify_images(train_path, cluster_number, k_means.labels_, train_label)
write_csv(train_label, k_means.labels_ + 1)
