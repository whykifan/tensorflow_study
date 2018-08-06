#coding=utf-8
import tensorflow as tf
import numpy as  np
import matplotlib.pyplot as plt
x_data = np.linspace(-1,1,200)[:,np.newaxis]  #生成一个新的维度
noise = np.random.normal(0,0.5,x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义第一层神经网络
w1 = tf.Variable(tf.random_normal([1,10]))
biases1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,w1) + biases1
L1 = tf.nn.tanh(Wx_plus_b_L1)

#第二层神经网络
w2 = tf.Variable(tf.random_normal([10,1]))
biases2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,w2) + biases2
L2 = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-L2))
#优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(2000):
		sess.run(train_step,feed_dict={x:x_data,y:y_data})
	prediction = sess.run(L2,feed_dict={x:x_data})
	print('the prediction value is {}'.format(prediction))
	plt.figure()
	plt.scatter(x_data,y_data)
	plt.plot(x_data,prediction,'r-',lw = 2)
	plt.show()
