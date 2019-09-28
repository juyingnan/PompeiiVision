import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import csv
# import random
from scipy import io as sio
import shutil
from skimage import io, transform
from tensorflow.contrib.factorization.python.ops import clustering_ops

tf.logging.set_verbosity(tf.logging.ERROR)
np.random.seed(0)
tf.set_random_seed(0)
batch_size = 128
full_length = 200
display_length = full_length
channel = 3
train_path = r'C:\Users\bunny\Desktop\test_20180919\unsupervised/'
train_image_count = 1000
emp = 1e-12

cluster_number = 4


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=5):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(emp + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(emp + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})


def read_img_random(path, file_names, as_gray=False, resize=None):
    imgs = list()
    # roman_label = ['I', 'II', 'III', 'IV']
    print('reading the images:%s' % path)
    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        img = io.imread(file_path, as_gray=as_gray)
        if resize is not None:
            img = transform.resize(img, resize, anti_aliasing=True)
        # io.imsave(file_path, img)
        if img.shape[-1] != 3 and not as_gray:
            print(file_path)
        imgs.append(img)

    return np.asarray(imgs, np.float32)


def next_batch(dataset, start_point, batch_size):
    if start_point >= len(dataset):
        return []
    elif start_point + batch_size <= len(dataset):
        return dataset[start_point: start_point + batch_size]
    else:
        return dataset[start_point:]


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def train(network_architecture, train_set, learning_rate=0.0001,
          batch_size=5, training_epochs=10, display_step=500):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = next_batch(train_set, i * batch_size, batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae


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


def predict_input_fn():
    return np.array(z_mu, np.float32)


def train_input_fn():
    data = tf.constant(z_mu, tf.float32)
    return data, None


def k_means(dataset):
    model = tf.contrib.learn.KMeansClustering(
        cluster_number,
        distance_metric=clustering_ops.SQUARED_EUCLIDEAN_DISTANCE,  # SQUARED_EUCLIDEAN_DISTANCE, COSINE_DISTANCE
        initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT
    )

    model.fit(input_fn=train_input_fn, steps=5000)

    print("--------------------")
    print("kmeans model: ", model)

    predictions = model.predict(input_fn=predict_input_fn, as_iterable=True)
    return predictions


def write_csv(img_name_list, cat_list, path='csv/ae_{0}.csv'.format(cluster_number)):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow((["NAME", "AE_CAT{0}".format(cluster_number)]))
        lines = []
        for i in range(len(img_name_list)):
            lines.append([img_name_list[i], cat_list[i]])
        writer.writerows(lines)


def read_csv(path):
    # init result
    file_names = list()
    styles = list()
    manual_features = list()

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        # skip header
        num_cols = len(next(reader))

        # assign file into sets
        for row in reader:
            filename = row[0]
            if len(filename) == 0:
                break
            file_names.append(filename)
            style = int(row[1])
            styles.append(style)
            manual_features.append(list())
            for i in range(2, num_cols):
                manual_features[-1].append(int(row[i]))

    return np.asarray(file_names, np.str_), np.asarray(styles, np.int8), np.asarray(manual_features, np.int8)


if __name__ == '__main__':
    csv_file_path = r'C:\Users\bunny\Desktop\Database_Revised.txt'
    file_name_list, style_list, manual_features_list = read_csv(csv_file_path)
    index_list = [list(style_list)[:i + 1].count(style_list[i]) for i in range(len(style_list))]

    image_root = r'C:\Users\bunny\Desktop\svd_test_500/'
    raw_pixel_list = read_img_random(image_root, file_name_list, resize=(full_length, full_length))
    if len(raw_pixel_list.shape) == 3:
        raw_pixel_1d_list = raw_pixel_list.reshape(
            (raw_pixel_list.shape[0], raw_pixel_list.shape[1] * raw_pixel_list.shape[2]))
    elif len(raw_pixel_list.shape) == 4:
        raw_pixel_1d_list = raw_pixel_list.reshape(
            (raw_pixel_list.shape[0], raw_pixel_list.shape[1] * raw_pixel_list.shape[2] * raw_pixel_list.shape[3]))
    else:
        raw_pixel_1d_list = []
    print(raw_pixel_1d_list.shape)
    n_samples = len(raw_pixel_1d_list)
    batch_size = int(n_samples / 1)

    network_architecture = \
        dict(n_hidden_recog_1=256,  # 1st layer encoder neurons
             n_hidden_recog_2=64,  # 2nd layer encoder neurons
             n_hidden_gener_1=64,  # 1st layer decoder neurons
             n_hidden_gener_2=256,  # 2nd layer decoder neurons
             n_input=full_length ** 2 * channel,  # MNIST data input (img shape: 28*28)
             n_z=32)  # dimensionality of latent space

    vae = train(network_architecture, raw_pixel_1d_list,
                learning_rate=0.0001,
                batch_size=batch_size,
                training_epochs=10000,
                display_step=100)

    test_sample = next_batch(raw_pixel_1d_list, 0, batch_size)
    x_reconstruct = vae.reconstruct(test_sample)

    plt.figure(figsize=(16, 20))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(test_sample[i].reshape(display_length, display_length, channel), vmin=0, vmax=1,
                   cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(display_length, display_length, channel), vmin=0, vmax=1,
                   cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show()

    x_sample = next_batch(raw_pixel_1d_list, 0, batch_size)
    z_mu = vae.transform(x_sample)

    vae.sess.close()

    row = len(z_mu)
    col = len(z_mu[0])
    print("[", row, "x", col, "] sized input")

    sio.savemat(file_name='../mat/autoencoder.mat',
                mdict={'feature_matrix': z_mu,
                       'label': style_list,
                       'file_name': file_name_list,
                       'index': index_list})

    # predictions = k_means(z_mu)
    #
    # index = 0
    # result_cat_list = []
    # for i in predictions:
    #     # print("[", z_mu[index], "] -> cluster_", i['cluster_idx'])
    #     result_cat_list.append(i['cluster_idx'])
    #     index = index + 1
    #
    # classify_images(train_path, cluster_number, result_cat_list, style_list)
    # for k in range(len(result_cat_list)):
    #     print(str(result_cat_list[k] + 1) + '\t' + style_list[k])
    # write_csv(style_list, [x + 1 for x in result_cat_list])
    #
    # import sys
    #
    # sys.exit("CUT")
    # network_architecture = \
    #     dict(n_hidden_recog_1=256,  # 1st layer encoder neurons
    #          n_hidden_recog_2=64,  # 2nd layer encoder neurons
    #          n_hidden_gener_1=64,  # 1st layer decoder neurons
    #          n_hidden_gener_2=256,  # 2nd layer decoder neurons
    #          n_input=full_length ** 2 * channel,  # MNIST data input (img shape: 28*28)
    #          n_z=2)  # dimensionality of latent space
    #
    # vae_2d = train(network_architecture, raw_pixel_1d_list, batch_size=batch_size, training_epochs=2000)
    #
    # x_sample = next_batch(raw_pixel_1d_list, 0, 75)
    # z_mu = vae_2d.transform(x_sample)
    # predictions = k_means(z_mu)
    #
    # index = 0
    # result_cat_list = []
    # for i in predictions:
    #     # print("[", d2_train_data[index], "] -> cluster_", i['cluster_idx'])
    #     result_cat_list.append(i['cluster_idx'])
    #     index = index + 1
    #
    # print(len(result_cat_list))
    #
    # plt.figure(figsize=(16, 12))
    # plt.scatter(z_mu[:, 0], z_mu[:, 1], c=result_cat_list)
    # plt.colorbar()
    # plt.grid()
    # plt.show()
    #
    # nx = ny = 20
    # x_values = np.linspace(-20, 20, nx)
    # y_values = np.linspace(-20, 25, ny)
    #
    # canvas = np.empty((display_length * ny, display_length * nx, channel))
    # for i, yi in enumerate(x_values):
    #     for j, xi in enumerate(y_values):
    #         z_mu = np.array([[xi, yi]] * vae.batch_size)
    #         x_mean = vae_2d.generate(z_mu)
    #         # print(x_mean)
    #         canvas[(nx - i - 1) * display_length:(nx - i) * display_length,
    #         j * display_length:(j + 1) * display_length] = \
    #             x_mean[0].reshape(display_length, display_length, channel)
    #
    # vae_2d.sess.close()
    # plt.figure(figsize=(16, 20))
    # Xi, Yi = np.meshgrid(x_values, y_values)
    # plt.imshow(canvas, origin="upper", cmap="gray")
    # plt.tight_layout()
    # plt.show()
