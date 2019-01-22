import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#使用循环神经网络实现手写数字识别,思想：将每个图片看成一行行相关的序列(28序列*28行)
#Rnn(LSTM)原理：https://www.jianshu.com/p/9dc9f41f0b29#
#以mnist_image行顺序就行输入，第一行为X_0,总输入为28个28维度的向量，many to one结构
mnist = input_data.read_data_sets('mnist/',one_hot=True)
train_X,train_Y,test_X,test_Y = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

#设置参数
lr = 0.001  #学习率
train_iters = 100000
batch_size = 128
#神经网络参数,序列长度28，步数28
n_inputs = 28
n_steps = 28
n_hidden_units = 128    #LTSM单元个数
n_classes = 10         #输出，类别数量
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])  #[batch_sizee,n_steps,n_inputs]
y = tf.placeholder(tf.float32,[None,n_classes])
#权重
weights = {
    #(28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    #(128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
    }
#偏置
biases = {
    #(128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

#Rnn模型
def Rnn(X,weights,biases):
    #将输入的数据转化为[128 batch*28steps,28 inputs],相当于把所有数据转化为一个数据
    X = tf.reshape(X,[-1,n_inputs])
    #隐藏层
    x_in = tf.matmul(X,weights['in']+biases['in'])
    x_in = tf.reshape(x_in,[-1,n_steps,n_hidden_units])
    #采用基本的LSTM循环神经网络单元，basicLSTMcell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    #初始化为零值，lstm的单元由两个部分组成(c_state,h_state)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    # dynamic_rnn 接收张量(batch, steps, inputs)或者(steps, batch, inputs)作为 X_in
    #outputs的维度为[batch_size,n_steps,n_hidden_units],保存了每一个step中cell的输出值
    #这里为many to one结构，所以只需要最后一个的输出值，outputs[:-1:]
    #result也可以写为 result = tf.layers.dense(inputs=outputs[:, -1, :], units=N_CLASSES),此为通用方法
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell,x_in,initial_state =init_state, time_major = False)
    #final_state[0]是cell state
    #final_state[1]是hidden state
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results
#定义损失函数与优化器，采用AdamOptimizer
pred = Rnn(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train = tf.train.AdamOptimizer(lr).minimize(cost)
#准确率
correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
#训练模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step*batch_size<train_iters:
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size,n_steps,n_inputs])
        sess.run(train,feed_dict={x:batch_x,y:batch_y})
        if step % 20==0:
            print(sess.run(correct,feed_dict={x:batch_x,y:batch_y}))
        step +=1
