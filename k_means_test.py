from sklearn import cluster
# from sklearn.neighbors import kneighbors_graph
# import random
import os
import shutil
from skimage import io, transform, color, exposure, segmentation, feature
import numpy as np
import csv

width = 250
height = 250
channel = 3
cluster_number = 4
color_bin_count = 20
largest_segment_count = 5
sift_feature_count = 12
upper_index = 0.35
lower_index = 0.2
lr_index = 0.2


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
        img = transform.resize(img, (width, height))
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


def write_csv(img_name_list, cat_list, path):
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
        result = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    elif len(data.shape) == 4:
        result = data.reshape((data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))
    else:
        result = []
    return result


def calculate_average_hue_saturation(img, h=True, s=True, v=True):
    img_hsv = color.rgb2hsv(img)
    img_h = img_hsv[:, :, 0]
    img_s = img_hsv[:, :, 1]
    img_v = img_hsv[:, :, 2]
    average_h = img_h.mean()
    average_s = img_s.mean()
    average_v = img_v.mean()
    result = []
    for pair in [(average_h, h), (average_s, s), (average_v, v)]:
        if pair[1]:
            result.append(pair[0])
    return result


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


def get_cropped_images(img, up=True, down=True, left=True, right=True, center=True):
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

    result = []
    for pair in [(up_img, up), (down_img, down), (left_img, left), (right_img, right), (central_img, center)]:
        if pair[1]:
            result.append(pair[0])

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


def calculate_seg_hsv(img_hsv, segments, max_seg_value_list, h=True, s=True, v=True):
    result = []
    for seg_value_pair in max_seg_value_list:
        segment_value, segment_value_count = seg_value_pair
        seg_h = 0
        seg_s = 0
        seg_v = 0
        for i in range(len(segments)):
            row = segments[i]
            for j in range(len(row)):
                if row[j] == segment_value:
                    seg_h += img_hsv[i, j, 0]
                    seg_s += img_hsv[i, j, 1]
                    seg_v += img_hsv[i, j, 2]
        seg_h /= segment_value_count
        seg_s /= segment_value_count
        seg_v /= segment_value_count

        for pair in [(seg_h, h), (seg_s, s), (seg_v, v)]:
            if pair[1]:
                result.append(pair[0])

    return result


def calculate_segmentation_mass_center(img):
    segments_fz = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    max_seg_value_list = calculate_n_max_seg_value(segments_fz, largest_segment_count)
    max_seg_mass_center_list = calculate_seg_mass_center(segments_fz, max_seg_value_list)
    # import matplotlib.pyplot as plt
    # plt.imshow(segments_fz)
    # plt.show()
    return max_seg_mass_center_list


def calculate_largest_n_sift(img, n):
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = color.rgb2gray(img)
    detector = feature.CENSURE()
    detector.detect(np.asarray(img, np.double))
    key_point_list = []
    # print(len(detector.scales))
    for i in range(len(detector.scales)):
        key_point_list.append((detector.keypoints[i], detector.scales[i]))
    sorted_key_point_list = sorted(key_point_list, reverse=True, key=lambda count: count[1])
    # result = [position for key_point in sorted_key_point_list[:n] for position in key_point[0]]
    result = [key_point[1] for key_point in sorted_key_point_list[:n]]
    while len(result) < n * 1:
        result.append(0)
    return result


def calculate_sift_count(img, x):
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = color.rgb2gray(img)
    detector = feature.CENSURE()
    detector.detect(np.asarray(img, np.double))
    key_point_count = 0
    # print(len(detector.scales))
    for i in range(len(detector.scales)):
        if detector.scales[i] >= x:
            key_point_count += 1
    # print(key_point_count)
    return [key_point_count]


def calculate_sift_distribution(img):
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = color.rgb2gray(img)
    detector = feature.CENSURE()
    detector.detect(img)
    key_point_distribution = [0, 0, 0, 0, 0]
    # print(len(detector.scales))
    for i in range(len(detector.scales)):
        if detector.keypoints[i][0] <= height * upper_index:
            key_point_distribution[0] += 1
        elif detector.keypoints[i][0] >= height * (1 - lower_index):
            key_point_distribution[1] += 1
        elif detector.keypoints[i][1] <= width * lr_index:
            key_point_distribution[2] += 1
        elif detector.keypoints[i][1] >= width * (1 - lr_index):
            key_point_distribution[3] += 1
        else:
            key_point_distribution[4] += 1
    # print(key_point_count)
    if len(detector.scales) > 0:
        key_point_distribution = [dist / len(detector.scales) for dist in key_point_distribution]
    return key_point_distribution


def get_global_color_features(data, whole_image_sample, frame_sample):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]

        if whole_image_sample:
            result[-1].extend(calculate_average_hue_saturation(img))
            result[-1].extend(calculate_hue_distribution(img))

        if frame_sample:
            cropped_imgs = get_cropped_images(img)
            for cropped_img in cropped_imgs:
                result[-1].extend(calculate_average_hue_saturation(cropped_img))
                result[-1].extend(calculate_hue_distribution(cropped_img))

    return result


def get_composition_features(data, whole_image_sample, frame_sample):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]

        if whole_image_sample:
            result[-1].extend(calculate_segmentation_mass_center(img))

        if frame_sample:
            cropped_imgs = get_cropped_images(img)
            for cropped_img in cropped_imgs:
                result[-1].extend(calculate_segmentation_mass_center(cropped_img))

    return result


def get_segment_color_features(data, whole_image_sample, frame_sample):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]

        if whole_image_sample:
            img_seg = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
            img_hsv = color.rgb2hsv(img)
            max_seg_value_list = calculate_n_max_seg_value(img_seg, largest_segment_count)
            result[-1].extend(calculate_seg_hsv(img_hsv, segments=img_seg, max_seg_value_list=max_seg_value_list,
                                                h=True, s=True, v=True))
        if frame_sample:
            cropped_imgs = get_cropped_images(img)
            for cropped_img in cropped_imgs:
                img_seg = segmentation.felzenszwalb(cropped_img, scale=100, sigma=0.5, min_size=50)
                img_hsv = color.rgb2hsv(cropped_img)
                max_seg_value_list = calculate_n_max_seg_value(img_seg, largest_segment_count)
                result[-1].extend(calculate_seg_hsv(img_hsv, segments=img_seg, max_seg_value_list=max_seg_value_list,
                                                    h=True, s=True, v=True))
    return result


def get_sift_features(data, whole_image_sample, frame_sample):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]

        if whole_image_sample:
            result[-1].extend(calculate_largest_n_sift(img, sift_feature_count))

        if frame_sample:
            cropped_imgs = get_cropped_images(img)
            for cropped_img in cropped_imgs:
                result[-1].extend(calculate_largest_n_sift(cropped_img, sift_feature_count))
    return result


def get_sift_count_features(data, whole_image_sample, frame_sample, x):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]

        if whole_image_sample:
            result[-1].extend(calculate_sift_count(img, x))

        if frame_sample:
            cropped_imgs = get_cropped_images(img)
            for cropped_img in cropped_imgs:
                result[-1].extend(calculate_sift_count(cropped_img, x))
    return result


def get_sift_distribution_features(data, whole_image_sample, frame_sample):
    result = []
    for i in range(len(data)):
        result.append([])
        img = data[i]

        if whole_image_sample:
            result[-1].extend(calculate_sift_distribution(img))

        if frame_sample:
            cropped_imgs = get_cropped_images(img)
            for cropped_img in cropped_imgs:
                result[-1].extend(calculate_sift_distribution(cropped_img))
    return result


def normalize_features(data, v_max=1.0, v_min=0.0):
    data_array = np.asarray(data, np.float32)
    mins = np.min(data_array, axis=0)
    maxs = np.max(data_array, axis=0)
    rng = maxs - mins
    result = v_max - ((v_max - v_min) * (maxs - data_array) / rng)
    return result


def get_features(data, whole_image_sample=True, frame_sample=False, global_color=True, composition=True,
                 segment_color=True, sift=True):
    result = []
    color_result = []
    composition_result = []
    segment_result = []
    sift_result = []
    for i in range(len(data)):
        result.append([])

    if global_color:
        color_result = get_global_color_features(data, whole_image_sample, frame_sample)
    if composition:
        composition_result = get_composition_features(data, whole_image_sample, frame_sample)
    if segment_color:
        segment_result = get_segment_color_features(data, whole_image_sample, frame_sample)
    if sift:
        sift_result = get_sift_features(data, whole_image_sample, frame_sample)

    for feature_result in [color_result, composition_result, segment_result, sift_result]:
        for i in range(len(feature_result)):
            result[i].extend(feature_result[i])

    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


def get_more_sift_features(data, sift_n_largest=True, sift_count=False, sift_large_count=False,
                           sift_distribution=False):
    result = []
    sift_n_largest_result = []
    sift_count_result = []
    sift_large_count_result = []
    sift_distribution_result = []
    for i in range(len(data)):
        result.append([])

    if sift_n_largest:
        sift_n_largest_result = get_sift_features(data, True, False)
    if sift_count:
        sift_count_result = get_sift_count_features(data, True, False, 0)
    if sift_large_count:
        sift_large_count_result = get_sift_count_features(data, True, False, 5)
    if sift_distribution:
        sift_distribution_result = get_sift_distribution_features(data, True, False)

    for feature_result in [sift_n_largest_result, sift_count_result, sift_large_count_result, sift_distribution_result]:
        for i in range(len(feature_result)):
            result[i].extend(feature_result[i])

    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


def k_means_clustering(data, path='', log=True, classify_folder=True):
    print("Compute K-means clustering...")
    k_means = cluster.KMeans(n_clusters=cluster_number)
    k_means.fit(data)
    if log:
        # print(k_means.labels_)
        # print(train_label)
        for k in range(len(k_means.labels_)):
            print(str(k_means.labels_[k] + 1) + '\t' + train_label[k])
    if classify_folder:
        classify_images(train_path, cluster_number, k_means.labels_, train_label)
    if path == '':
        write_csv(train_label, k_means.labels_ + 1, path='csv/kmeans_{0}.csv'.format(cluster_number))
    else:
        write_csv(train_label, k_means.labels_ + 1, path=path)


def hierarchical_clustering(data, path='', log=True, classify_folder=True):
    print("Compute unstructured hierarchical clustering...")
    # connectivity = kneighbors_graph(data, n_neighbors=5, include_self=False)
    ward = cluster.AgglomerativeClustering(n_clusters=cluster_number,  # connectivity=connectivity,
                                           linkage='ward').fit(data)
    if log:
        for k in range(len(ward.labels_)):
            print(str(ward.labels_[k] + 1) + '\t' + train_label[k])
    if classify_folder:
        classify_images(train_path, cluster_number, ward.labels_, train_label)
    if path == '':
        write_csv(train_label, ward.labels_ + 1, path='csv/hierarchical_{0}.csv'.format(cluster_number))
    else:
        write_csv(train_label, ward.labels_ + 1, path=path)


def loop_run(data):
    for whole in [True, False]:
        for frame in [True, False]:
            if whole or frame:
                for gc in [True, False]:
                    for co in [True, False]:
                        for sc in [True, False]:
                            for si in [True, False]:
                                if gc or co or sc or si:
                                    d2_data = get_features(data, whole_image_sample=whole, frame_sample=frame,
                                                           global_color=gc, composition=co,
                                                           segment_color=sc, sift=si)
                                    str_format = 100000 * (1 if whole else 0) + 10000 * (1 if frame else 0) + 1000 * (
                                        1 if gc else 0) + 100 * (1 if co else 0) + 10 * (1 if sc else 0) + 1 * (
                                                     1 if si else 0)
                                    k_means_clustering(d2_data,
                                                       path='csv/kmeans_{0}_{1:06d}.csv'.format(cluster_number,
                                                                                                str_format),
                                                       log=False, classify_folder=False)
                                    hierarchical_clustering(d2_data,
                                                            'csv/hierarchical_{0}_{1:06d}.csv'.format(cluster_number,
                                                                                                      str_format),
                                                            log=False, classify_folder=False)
                                    print("{0:06d} done".format(str_format))


if __name__ == '__main__':
    train_path = r'C:\Users\bunny\Desktop\test_20180919\unsupervised/'
    train_image_count = 1000
    train_data, train_label = read_img_random(train_path, train_image_count)

    # K-Means
    # loop_run(train_data)

    d2_train_data = get_features(train_data, whole_image_sample=True, frame_sample=False, global_color=False,
                                 composition=False, segment_color=False, sift=True)
    # d2_train_data = get_more_sift_features(train_data, sift_n_largest=True, sift_count=False, sift_large_count=False,
    #                                        sift_distribution=False)
    # d2_train_data = get_raw_pixel_features(train_data)

    k_means_clustering(d2_train_data)
    hierarchical_clustering(d2_train_data)

    # test
    # for t in range(100):
    #     k_means_clustering(d2_train_data, 'csv/kmeans_{0}_{1:03d}.csv'.format(cluster_number, t + 1), False, False)
    #     hierarchical_clustering(d2_train_data, 'csv/hierarchical_{0}_{1:03d}.csv'.format(cluster_number, t + 1),
    #                         False, False)
