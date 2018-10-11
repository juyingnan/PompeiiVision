from skimage.feature import daisy
from skimage import color, io, transform
# import matplotlib.pyplot as plt
import os
import csv
import shutil
import math
from sklearn import cluster
import numpy as np
import ClusterMatching
from skimage.feature import hog
from skimage.feature import blob_log
from skimage.feature import ORB
from skimage.util.shape import view_as_windows
from skimage.util import montage
from scipy.cluster.vq import kmeans2
from scipy import ndimage as ndi
from skimage.feature import shape_index

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
        img = io.imread(im, as_gray=False)
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
    # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_daisy.html#sphx-glr-auto-examples-features-detection-plot-daisy-py
    result = []
    for i in range(len(data)):
        img = data[i]
        img = color.rgb2gray(img)
        descs, descs_img = daisy(img, step=180, radius=58, rings=2, histograms=6,
                                 orientations=8, visualize=True)
        result.append(descs.flatten())
    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


def get_histogram_features(data):
    # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html#sphx-glr-auto-examples-features-detection-plot-hog-py
    result = []
    for i in range(len(data)):
        img = data[i]
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)
        # plt.imshow(hog_image)
        # plt.show()
        result.append(fd.flatten())
    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


def get_blob_features(data):
    # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html#sphx-glr-auto-examples-features-detection-plot-blob-py
    result = []
    for i in range(len(data)):
        img = data[i]
        image_gray = color.rgb2gray(img)
        blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
        blobs_log[:, 2] = blobs_log[:, 2] * math.sqrt(2)

        # plt.imshow(hog_image)
        # plt.show()
        blobs_log.view('i8,i8,i8').sort(order=['f2'], axis=0, )
        result.append([x[2] for x in blobs_log[-10:]])
    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


def get_orb_features(data):
    # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_orb.html#sphx-glr-auto-examples-features-detection-plot-orb-py
    result = []
    for i in range(len(data)):
        img = data[i]
        image_gray = color.rgb2gray(img)

        descriptor_extractor = ORB(n_keypoints=200)

        descriptor_extractor.detect_and_extract(np.asarray(image_gray, np.double))
        keypoints = descriptor_extractor.keypoints
        # descriptors = descriptor_extractor.descriptors

        # plt.imshow(hog_image)
        # plt.show()
        result.append(keypoints.flatten())
    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


def get_filterbank_features(data):
    # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabors_from_astronaut.html#sphx-glr-auto-examples-features-detection-plot-gabors-from-astronaut-py
    # -- filterbank1 on original image
    result = []
    for i in range(len(data)):
        img = data[i]
        image_gray = color.rgb2gray(img)
        patch_shape = 8, 8
        n_filters = 49
        patches1 = view_as_windows(image_gray, patch_shape)
        patches1 = patches1.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
        fb1, _ = kmeans2(patches1, n_filters, minit='points')
        fb1 = fb1.reshape((-1,) + patch_shape)
        fb1_montage = montage(fb1, rescale_intensity=True)

        # plt.imshow(hog_image)
        # plt.show()
        result.append(fb1_montage.flatten())
    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


def get_filterbank2_features(data):
    # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabors_from_astronaut.html#sphx-glr-auto-examples-features-detection-plot-gabors-from-astronaut-py
    # -- filterbank2 LGN-like image
    result = []
    for i in range(len(data)):
        img = data[i]
        image_gray = color.rgb2gray(img)
        image_gray = ndi.gaussian_filter(image_gray, .5) - ndi.gaussian_filter(image_gray, 1)
        patch_shape = 8, 8
        n_filters = 49
        patches1 = view_as_windows(image_gray, patch_shape)
        patches1 = patches1.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
        fb1, _ = kmeans2(patches1, n_filters, minit='points')
        fb1 = fb1.reshape((-1,) + patch_shape)
        fb1_montage = montage(fb1, rescale_intensity=True)

        # plt.imshow(hog_image)
        # plt.show()
        result.append(fb1_montage.flatten())
    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


def get_shape_index_features(data, size=10):
    # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_shape_index.html#sphx-glr-auto-examples-features-detection-plot-shape-index-py
    # -- filterbank1 on original image
    result = []
    for i in range(len(data)):
        img = data[i]
        image_gray = color.rgb2gray(img)
        s = shape_index(image_gray)

        # In this example we want to detect 'spherical caps',
        # so we threshold the shape index map to
        # find points which are 'spherical caps' (~1)

        target = 1
        delta = 0.05

        point_y, point_x = np.where(np.abs(s - target) < delta)
        point_z = image_gray[point_y, point_x]

        # The shape index map relentlessly produces the shape, even that of noise.
        # In order to reduce the impact of noise, we apply a Gaussian filter to it,
        # and show the results once in

        # s_smooth = ndi.gaussian_filter(s, sigma=0.5)

        # point_y_s, point_x_s = np.where(np.abs(s_smooth - target) < delta)
        # point_z_s = image_gray[point_y_s, point_x_s]

        # plt.imshow(hog_image)
        # plt.show()
        point_z.sort()
        result.append(point_z[-size:])
    result = normalize_features(result, v_max=1.0, v_min=0.0)
    return result


train_path = r'C:\Users\bunny\Desktop\test_20180919\unsupervised/'
train_image_count = 1000
train_data, train_label = read_img_random(train_path, train_image_count)
np.seterr(all='ignore')
# K-Means
# loop_run(train_data)

# d2_train_data = get_dense_daisy_features(train_data)
# d2_train_data = get_histogram_features(train_data)
# d2_train_data = get_blob_features(train_data)
# d2_train_data = get_orb_features(train_data)
# d2_train_data = get_filterbank_features(train_data)
# d2_train_data = get_filterbank2_features(train_data)
d2_train_data = get_shape_index_features(train_data)

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
