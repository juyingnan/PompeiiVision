from skimage.feature import daisy
from skimage import color, io, transform
import matplotlib.pyplot as plt
import os
import csv
import shutil
from sklearn import cluster
import numpy as np
import ClusterMatching

width = 500
height = 500
channel = 3
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
        img = io.imread(im, as_gray=True)
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = transform.resize(img, (width, height))
        imgs.append(img)
        labels.append(file_name)
        if count % 1 == 0:
            print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
    print('\r', end='')
    return np.asarray(imgs, np.float32), np.asarray(labels, np.str_)


def write_csv(img_name_list, cat_list, path):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow((["NAME", "KMEANS_CAT{0}".format(cluster_number)]))
        lines = []
        for i in range(len(img_name_list)):
            lines.append([img_name_list[i], cat_list[i]])
        writer.writerows(lines)


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


def hierarchical_clustering(data, path='', log=False, classify_folder=False):
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


def normalize_features(data, v_max=1.0, v_min=0.0):
    data_array = np.asarray(data, np.float32)
    mins = np.min(data_array, axis=0)
    maxs = np.max(data_array, axis=0)
    rng = maxs - mins
    result = v_max - ((v_max - v_min) * (maxs - data_array) / rng)
    return result


def get_dense_daisy_features(data):
    result = []
    for i in range(len(data)):
        img = data[i]
        # img = color.rgb2gray(img)
        descs, descs_img = daisy(img, step=180, radius=58, rings=2, histograms=6,
                                 orientations=8, visualize=True)
        result.append(descs.flatten())
    return result


train_path = r'C:\Users\bunny\Desktop\test_20180919\unsupervised/'
train_image_count = 1000
train_data, train_label = read_img_random(train_path, train_image_count)

# K-Means
# loop_run(train_data)

d2_train_data = get_dense_daisy_features(train_data)

csv_path = 'csv/dense_daisy_hierarchical_{0}.csv'.format(cluster_number)
hierarchical_clustering(d2_train_data, path=csv_path)

# assign sets and lists
# human_cat, kmeans_cat, ae_cat = read_csv(csv_path)
human_cat = ClusterMatching.read_csv('csv/human_4.csv')
hierarchical_cat = ClusterMatching.read_csv(csv_path)

# find best match
human_hierarchical_match = ClusterMatching.find_best_match_cat4(human_cat, hierarchical_cat)

# find route from best match
print('****************')
print('human & hierarchical')
ClusterMatching.find_route(human_cat, hierarchical_cat, human_hierarchical_match, "Human", "Hierarchical",
                           sankeymatic_output_format=True)
print('****************')
