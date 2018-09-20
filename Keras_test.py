import os
import random
from skimage import io, transform
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Sequential
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


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


w = 500
h = 500
c = 3
train_image_count = 1000
val_image_count = train_image_count / 10
test_image_count = train_image_count / 10
input_shape = (w, h, c)
learning_rate = 0.0001
regularization_rate = 0.00001
category_count = 4
n_epoch = 100
mini_batch_size = 25
# data set path
train_path = r'C:\Users\bunny\Desktop\test_20180919\supervised\TRAIN/'
val_path = r'C:\Users\bunny\Desktop\test_20180919\supervised\VAL/'
test_path = r'C:\Users\bunny\Desktop\test_20180919\supervised\TEST/'

model = Sequential()

# Layer 1
model.add(Conv2D(32,
                 kernel_size=(11, 11),
                 strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

# Layer 2
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

# Layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten
model.add(Flatten(input_shape=input_shape))

# fc layers
model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dense(category_count, activation='softmax', kernel_regularizer=regularizers.l2(regularization_rate)))

# read image
train_data, train_label = read_img_random(train_path, train_image_count)
val_data, val_label = read_img_random(val_path, val_image_count)
test_data, test_label = read_img_random(test_path, test_image_count)

x_train = train_data
y_train = train_label
x_val = val_data
y_val = val_label
x_test = test_data
y_test = test_label

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
# x_train = x_train.reshape(x_train.shape[0], w, h, c)
# x_val = x_val.reshape(x_val.shape[0], w, h, c)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, category_count)
y_val = keras.utils.to_categorical(y_val, category_count)
y_test = keras.utils.to_categorical(y_test, category_count)

# train
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# train
history = AccuracyHistory()
model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.SGD(lr=0.01),
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=mini_batch_size,
          epochs=n_epoch,
          verbose=2,
          validation_data=(x_val, y_val),
          callbacks=[history])
model.save_weights(train_path + '/model_weight.h5')
model.save(train_path + '/model.h5')
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, n_epoch + 1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
