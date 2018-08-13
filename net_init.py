#coding=utf-8
import tensorflow as tf

'''
tensorflow的一些比较好的参数初始化模板，以三层神经网络为例
'''

#初始化权重
def get_weight(shape,regularizer=None):
	w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
	'''
	#判断是否使用正则化以提高模型的泛化能力，如果使用正则化，需要使用下面的根据规则使用以下损失函数
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_, 1))  #交叉熵
	cem = tf.reduce_mean(ce) 							#正常损失函数
	loss = cem + tf.add_n(tf.get_collection('losses'))  #正则化损失函数的公式
	'''
	if regularizer!=None:
		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

#初始化偏置
def get_bias(shape):
	b = tf.Variable(tf.constant(0.01,shape=shape))
	return b

#初始化前向传播网络
def forward(x_data,regularizer):
	#多层神经网络权重和偏置的一种比较好的初始化方法
	#有利于代码的整洁与修改
	W = {
		'w1':get_weight(shape1,regularizer1)
		'w2':get_weight(shape2,regularizer2)
		#'w3'......
		'out':get_weight(shape3,regularizer3)
		}
	B = {
		'b1':get_bias(shape1)
		'b2':get_bias(shape2)
		#'b3'....
		'out':get_bias(shape3)
		}
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x_data,W['w1']),B['b1']))
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,W['w2']),B['b2']))
	out = tf.nn.sigmoid(tf.add(tf.matmul(layer2,W['out']),B['out']))

'''前向神经网络定义完成以后便可以进行损失函数的定义，以及使用相应的优化器，如使用正则化，则优化器应传入相应的正则化损失函数'''


#另一种至初始化神经网络层的方法
W = {
		'w1':tf.Variable(tf.random_normal(shape1，stddev=stddev))
		'w2':tf.Variable(tf.random_normal(shape2，stddev=stddev))
		#'w3'......
		'out':tf.Variable(tf.random_normal(shape3，stddev=stddev))
	}
B = {
		'b1':tf.Variable(tf.constant(0.01,shape=shape1))
		'b2':tf.Variable(tf.constant(0.01,shape=shape2))
		#'b3'....
		'out':tf.Variable(tf.constant(0.01,shape=shape3))
	}
def mutilayer_perceptron(_X,_W,_B):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x_data,W['w1']),B['b1']))
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,W['w2']),B['b2']))
	return tf.nn.sigmoid(tf.add(tf.matmul(layer2,W['out']),B['out']))