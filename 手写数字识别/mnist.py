from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''使用逻辑回归的方法进行分类
#读取数据
mnist = input_data.read_data_sets('mnist/',one_hot=True)
#每个手写数字图像的大小为28*28=784，分类数量为10
data = tf.placeholder(tf.float32,[None,784])
#权重一般使用正太随机初始化，偏置一般使用0
w1 = tf.Variable(tf.random_normal([784,10]))
b1 = tf.Variable(tf.zeros([10]))
y = tf.add(tf.matmul(data,w1),b1)
y_real = tf.placeholder(tf.float32,[None,10])
#使用交叉熵来计算损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_real,logits=y))
#使用SGD来进行优化
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#准确率,每一行的预测是否相等，并取均值，判断准确率,要在测试集上进行判断
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_real,1)),tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #进行训练
    for i in range(1000):
        batch_x,batch_y = mnist.train.next_batch(200)
        sess.run(train_step,feed_dict={data:batch_x,y_real:batch_y})
        if i%50 == 0:
            print(sess.run(accuracy,feed_dict={data:mnist.test.images,y_real:mnist.test.labels}))
'''
'''使用CNN进行手势识别分类'''
mnist = input_data.read_data_sets('mnist/',one_hot=True)
train_X,train_Y,test_X,test_Y = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
#为了不损失图像的信息，将原来一维的数据展开为二维
# [-1,28,28,1] -1表示不考虑图像数量，[28，28]图像大小，灰色图像通道数为1
train_X = train_X.reshape([-1,28,28,1])
test_X = test_X.reshape([-1,28,28,1])   #使用tf.reshape进行转至的话需要会话进行运行

data = tf.placeholder(tf.float32,[None,28,28,1])
y_real = tf.placeholder(tf.float32,[None,10])   #数据标签的形状并没有更改
#drop_out概率
keep_prop_5 = tf.placeholder(tf.float32)
keep_prop_75 = tf.placeholder(tf.float32)
#训练模型保存
Dir = 'train_data/'

# #用于使用tensorboard查看数据变化
# def variable_summaries(var):
#   with tf.name_scope('summaries'):
#     mean = tf.reduce_mean(var)  #均值
#     tf.summary.scalar('mean', mean)
#     with tf.name_scope('stddev'):
#       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))    #均方误差
#     tf.summary.scalar('stddev', stddev)
#     tf.summary.scalar('max', tf.reduce_max(var))
#     tf.summary.scalar('min', tf.reduce_min(var))
#     tf.summary.histogram('histogram', var)

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))
def init_bias(shape):
    return tf.Variable(tf.zeros(shape))

#定义网络结构，使用3层卷积层，3层池化层，1层全连接层，输出层，每层使用drop_out
def model(input_data):
    w1 = init_weight([3,3,1,32])    #使用卷积核大小为3*3，输入深度为1，输出深度为32,相当于提取32种特征
    b1 = init_bias([32])            #偏置与深度保持一致
    w2 = init_weight([3,3,32,64])   #本层网络输入深度应与上层网络输出深度一致
    b2 = init_bias([64])
    w3 = init_weight(([3,3,64,128]))
    b3 = init_bias([128])
    w_f = init_weight([128*4*4,625])#全连接层
    b_f = init_bias([625])
    w_o = init_weight([625,10])     #输出层
    b_0 = init_bias([10])
    #第一层
    l1 = tf.nn.relu(tf.nn.conv2d(input_data,w1,strides=[1,1,1,1],padding='SAME')+b1)
    l1_p = tf.nn.max_pool(l1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')        #如果使用池化步长为2,相当于已经进行了drop_out?
    l1_o = tf.nn.dropout(l1_p,keep_prop_5)                    #使用了一次最大池化，大小变为[-1,14,14,32]
    #第二层
    l2 = tf.nn.relu(tf.nn.conv2d(l1_o,w2,strides=[1,1,1,1],padding='SAME')+b2)
    l2_p = tf.nn.max_pool(l2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    l2_o = tf.nn.dropout(l2_p,keep_prop_5)                    #使用了一次最大池化，大小变为[-1,7,7,64]
    #第三层
    l3 = tf.nn.relu(tf.nn.conv2d(l2_o,w3,strides=[1,1,1,1],padding='SAME')+b3)
    l3_p = tf.nn.max_pool(l3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    l3_o = tf.nn.dropout(l3_p,keep_prop_5)           #使用了一次最大池化，大小变为[-1,4,4,128]
    #为与全连接层进行连接，将特征图变为一维向量
    l3_f = tf.reshape(l3_o,[-1,128*4*4])
    #全连接层
    l4 = tf.nn.relu(tf.add(tf.matmul(l3_f,w_f),b_f))
    l4_o = tf.nn.dropout(l4,keep_prop_75)
    #输出层
    out = tf.add(tf.matmul(l4_o,w_o),b_0)
    #返回预测值
    return out

y = model(data)
#使用交叉熵来计算损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_real))
train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(loss)
#得到精度
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_real,1)),tf.float32))
batch_size = 128
test_size = 256  #数据过大，不能一次性读取完成
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        #zip函数返回一个以元组为元素的列表，第i个元素包含每个参数序列的第i个元素
        #例如：l1 = [1,2,3,4,5]，l2 = [11,22,33,44,55]，l3 = zip(l1,l2)
        #l3 = [(1, 11),(2, 22).(3, 33),(4, 44) ,(5, 55)]
        train_batch = zip(range(0,len(train_X),batch_size),range(batch_size,len(train_X)+1,batch_size))
        for start,end in train_batch:
            sess.run(train_step,feed_dict={data:train_X[start:end],y_real:train_Y[start:end],keep_prop_5:0.5,keep_prop_75:0.75})
        if i % 10 == 0:
            test_indices = np.arange(len(test_X))  # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            print(sess.run(accuracy,feed_dict={data:test_X[test_indices],y_real:test_Y[test_indices],keep_prop_5:1.0,keep_prop_75:1.0}))