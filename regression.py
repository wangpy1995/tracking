import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data/new_10.csv')


def regress(x_data, y_data, iter: int):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 1])  # 任意行1列矩阵
    y = tf.placeholder(tf.float32, [None, 1])  # 任意行1列矩阵

    # 定义神经网络中间层
    Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
    biases_L1 = tf.Variable(tf.zeros([1, 10], tf.float32))
    W_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
    L1 = tf.nn.tanh(W_plus_b_L1)

    # 定义神经网络输出层
    Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
    biases_L2 = tf.Variable(tf.zeros([1, 1]))
    W_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
    output = tf.tanh(W_plus_b_L2)

    # 二次代价函数
    loss = tf.reduce_mean(tf.square(y - output))

    # 使用梯度下降法训练
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(iter):
            sess.run(train_step, feed_dict={x: x_data, y: y_data})
        # 获得预测值
        output_value = sess.run(output, feed_dict={x: x_data})
        # 画图
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, output_value, 'b-', lw=5)
        return output_value
