# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2019/5/11
import numpy as np
np.random.seed(1337)  # 设置随机生成算法的初始值
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

X = np.linspace(-1, 1, 200)  # 随机生成(-1,1)的200个一位数组
np.random.shuffle(X)  # 随机打乱数据
Y = 0.5*X + 2 + np.random.normal(0, 0.05, (200, ))  # normal生成200个高斯分布,loc(此概率分布的均值),scale(此概率的标准差)

plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]  # 前160个点
X_test, Y_test = X[160:], Y[160:]   # 后40个点
# 建立模型
model = Sequential()
# 添加神经层
model.add(Dense(input_dim=1, units=1))  # 因为该线性方程是1维的,所以输出也是一维
# 激活模型
model.compile(loss='mse', optimizer='sgd')  # 参数中，误差函数用的是 mse 均方误差；优化器用的是 sgd 随机梯度下降法。

# 训练模型
print('Training ----------------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost:', cost)

# 检验模型
'''用到的函数是 model.evaluate，输入测试集的x和y， 
输出 cost，weights 和 biases。其中 weights 和 biases 是取在模型的第一层 model.layers[0] 学习到的参数。
从学习到的结果你可以看到, weights 比较接近0.5，bias 接近 2。'''
print('\nTesting ---------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# 可视化结果

Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)  # 绘制散点图
plt.plot(X_test, Y_pred)  # 绘制线图
plt.show()


