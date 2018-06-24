import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
from skimage import io, transform
import numpy as np

w = 200
h = 200
c = 3


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
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
            if count % 100 == 0:
                print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
        print('\r', end='')
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


# 设置训练超参数
learning_rate = 0.01  # 学习速率
training_epochs = 10  # 训练轮数
batch_size = 256  # 训练批次大小
display_step = 1  # 显示间隔
examples_to_show = 10  # 表示从测试集中选择十张图片去验证自动编码器的结果
n_input = 784  # 数据的特征值个数

# 输入数据，不需要标记
X = tf.placeholder("float", [None, n_input])

# 用字典的方式存储各隐藏层的参数   网络参数
n_hidden_1 = 256  # 第一编码层神经元个数256，也是特征值个数
n_hidden_2 = 128  # 第二编码层神经元个数128，也是特征值个数

# 权重和偏置的变化在编码层和解码层顺序是相逆的
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# 定义有压缩函数，每一层结构都是 xW + b
# 构建编码器
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# 定义解压函数
# 构建解码器
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# 构建模型
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# 预测
y_pred = decoder_op  # 得出预测值
y_true = X  # 输入值 即得出真实值

# 定义代价函数和优化器
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))  # 最小二乘法 平方差取平均
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # 优化器采用 AdamOptimizer 或者 RMSPropOptimizer

# 训练数据和评估模型
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    # 检测当前TF版本
    init = tf.global_variables_initializer()
    sess.run(init)

    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
    total_batch = int(mnist.train.num_examples / batch_size)  # 总批数
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")

    # 对测试集应用训练好的自动编码网络
    encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # 比较测试集原始图片和自动编码网络的重建结果
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  # 测试集
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))  # 重建结果
    plt.show()
    # plt.draw()
    # plt.waitforbuttonpress()

print('Finish!')
