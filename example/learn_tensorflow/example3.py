# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2019/5/11
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 添加层
'''然后定义添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数，
我们设定默认的激励函数是None。'''


def add_layer(inputs, in_size, out_size, activation_function=None):

    '''
    接下来，我们开始定义weights和biases。因为在生成初始参数时，
    随机变量(normal distribution)会比全部为0要好很多，
    所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。
    '''
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    '''在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。'''
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    '''下面，我们定义Wx_plus_b, 即神经网络未激活的值。其中，tf.matmul()是矩阵的乘法。'''
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    '''当activation_function——激励函数为None时，输出就是当前的预测值——Wx_plus_b，不为None时，
    就把Wx_plus_b传到activation_function()函数中得到输出。
    '''
    if activation_function is None:
        out_size = Wx_plus_b
    else:
        out_size = activation_function(Wx_plus_b)

    return out_size


# 导入数据
'''导入数据构建所需的数据。 这里的x_data和y_data并不是严格的一元二次函数的关系，
因为我们多加了一个noise,这样看起来会更像真实情况。'''
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

'''利用占位符定义我们所需的神经网络的输入。 tf.placeholder()就是代表占位符，
这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1。'''
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
'''接下来，我们就可以开始定义神经层了。 通常神经层都包括输入层、隐藏层和输出层。
这里的输入层只有一个属性， 所以我们就只有一个输入；隐藏层我们可以自己假设，这里我们假设隐藏层有10个神经元； 
输出层和输入层的结构是一样的，所以我们的输出层也是只有一层。 
所以，我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。'''

# 搭建网络
'''我们开始定义隐藏层,利用之前的add_layer()函数，这里使用 Tensorflow 自带的激励函数tf.nn.relu。'''
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
'''接着，定义输出层。此时的输入就是隐藏层的输出——l1，输入有10层（隐藏层的输出层），输出有1层。'''
prediction = add_layer(l1, 10, 1, activation_function=None)
'''计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。'''
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
'''接下来，是很关键的一步，如何让机器学习提升它的准确率。
tf.train.GradientDescentOptimizer()中的值通常都小于1，这里取的是0.1，代表以0.1的效率来最小化误差loss。'''
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化
init = tf.global_variables_initializer()

'''定义Session，并用 Session 来执行 init 初始化步骤。 （注意：在tensorflow中，只有session.run()才会执行我们定义的运算。）'''
sess = tf.Session()
sess.run(init)

# matplotlib可视化
'''构建图形，用散点图描述真实数据之间的关系。 （注意：plt.ion()用于连续显示。）'''
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# 训练
'''下面，让机器开始学习。

比如这里，我们让机器学习1000次。机器学习的内容是train_step, 
用 Session 来 run 每一次 training 的数据，逐步提升神经网络的预测准确性。 
(注意：当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。)'''

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(1)
