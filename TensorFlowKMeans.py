import tensorflow as tf
import numpy as np
import pylab as pl
import os
import shutil
from skimage import io
from tensorflow.contrib.factorization.python.ops import clustering_ops

from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------
# variables
classes = 4  # define number of clusters
display3D = True


# -----------------------------------------
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
        # img = transform.resize(img, (w, h))
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


# dataset
train_path = r'C:\Users\bunny\Desktop\km\500/'
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

row = len(d2_train_data)
col = len(d2_train_data[0])

print("[", row, "x", col, "] sized input")

if display3D is False:
    for i in range(row):
        pl.scatter(d2_train_data[i][0], d2_train_data[i][1], c='black')
    pl.show()

# -----------------------------------------

model = tf.contrib.learn.KMeansClustering(
    classes,
    distance_metric=clustering_ops.SQUARED_EUCLIDEAN_DISTANCE,  # SQUARED_EUCLIDEAN_DISTANCE, COSINE_DISTANCE
    initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT
)


# -----------------------------------------

def train_input_fn():
    data = tf.constant(d2_train_data, tf.float32)
    return data, None


model.fit(input_fn=train_input_fn, steps=5000)

print("--------------------")
print("kmeans model: ", model)


def predict_input_fn():
    return np.array(d2_train_data, np.float32)


predictions = model.predict(input_fn=predict_input_fn, as_iterable=True)

colors = ['orange', 'red', 'blue', 'green']

print("--------------------")

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')

index = 0
result_cat_list = []
for i in predictions:
    print("[", d2_train_data[index], "] -> cluster_", i['cluster_idx'])
    result_cat_list.append(i['cluster_idx'])
    color_aver = np.mean(train_data[index], axis=(0, 1))
    if display3D is False:
        pl.scatter(d2_train_data[index][0], d2_train_data[index][1], c=colors[i['cluster_idx']])  # 2d graph
    if display3D is True:
        ax.scatter(color_aver[0], color_aver[1], color_aver[2], c=colors[i['cluster_idx']])  # 3d graph

    index = index + 1

pl.show()
classify_images(train_path, classes, result_cat_list, train_label)
