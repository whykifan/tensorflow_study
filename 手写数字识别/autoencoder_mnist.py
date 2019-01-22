import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
'''无监督学习，自编码神经网络实现手写数字识别'''

mnist = input_data.read_data_sets('mnist/',one_hot=True)

#参数设置
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_steps = 1

example_to_show = 10  #从测试集选择10张图片验证结果

#网络参数
n_hidden1 = 256
n_hidden2 = 128
n_inputs = 784

X = tf.placeholder(tf.float32,[None,n_inputs])

weights = {
    #压缩
    'encoder_w1':tf.Variable(tf.random_normal([n_inputs,n_hidden1])),
    'encoder_w2':tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    #解码
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden2, n_hidden1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden1, n_inputs]))
}
biases = {
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden2])),
    'decoder_b1':tf.Variable(tf.random_normal([n_hidden1])),
    'decoder_b2':tf.Variable(tf.random_normal([n_inputs])),
}

#定义网络结构
def autoencoder_net(x):
    #使用sigmoid激活函数,压缩过程
    encoder_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_w1']),biases['encoder_b1']))
    encoder_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer1,weights['encoder_w2']),biases['encoder_b2']))
    decoder_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer2, weights['decoder_h1']), biases['decoder_b1']))
    decoder_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_layer1, weights['decoder_h2']), biases['decoder_b2']))
    return decoder_layer2

cost = tf.reduce_mean(tf.pow(X-autoencoder_net(X),2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples / batch_size)
    # 开始训练
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    # 每一轮，打印出一次损失值
        if epoch % display_steps == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    encode_decode = sess.run(autoencoder_net(X), feed_dict={X: mnist.test.images[:example_to_show]})
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(example_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  # 测试集
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))  # 重建结果
    f.show()
    plt.draw()
    plt.waitforbuttonpress()