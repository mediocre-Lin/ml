#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/3/30 12:32
# Tool :PyCharm
# 多元线性函数 三种梯度下降算法
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from Utils import GD, SGD, mini_batch_SGD


def fn1(x, theta):
    return theta[0] * x[:, 0] + theta[1] * x[:, 1]


def fn2(x, theta):
    return theta[0] * x[:, 0] + theta[1] * x[:, 1] + theta[2] * x[:, 2]

#
# # 生成 y=10x + 5 + noise 数据
# x = np.array([[1, i] for i in range(50)])
# y = np.array(50 * np.random.rand(50) + fn1(x, [5, 10]))
# print('gd')
# gd = GD(x, y, fn1, lr=1e-3, verbose=False)
# gd_theta,gd_theta_list = gd.step()
# print('sgd')
# sgd = SGD(x, y, fn1, lr=1e-3, verbose=False)
# sgd_theta,sgd_theta_list= sgd.step()
# print('mini_batch_sgd')
# mini_batch_sgd = mini_batch_SGD(x, y, fn1, lr=1e-3, verbose=False, batch=8)
# mini_batch_sgd_theta,mini_batch_sgd_list= mini_batch_sgd.step()
# print(gd_theta, sgd_theta, mini_batch_sgd_theta)
#
#
# plt.figure(1)
# plt.scatter(x[:, 1], y)
# plt.title('y = 10x+5+noise')
# plt.plot(range(50), fn1(x, gd_theta), color='black')
# plt.plot(range(50), fn1(x, sgd_theta), color='yellow')
# plt.plot(range(50), fn1(x, mini_batch_sgd_theta), color='deeppink')
# plt.legend(['gb', 'sgd', 'mini_batch_sgd'])
# plt.savefig('gd1.png', dpi=200)
#
# plt.figure(2)
# plt.subplot(2,2,1)
# plt.plot(gd_theta_list[:,0], gd_theta_list[:,1], color='black')
# plt.title('gb')
# plt.subplot(2,2,2)
# plt.plot(sgd_theta_list[:,0], sgd_theta_list[:,1], color='yellow')
# plt.title('sgd')
# plt.subplot(2,2,3)
# plt.plot(mini_batch_sgd_list[:,0], mini_batch_sgd_list[:,1], color='deeppink')
# plt.title('mini_batch_sgd')
# plt.savefig('gd1_theta.png', dpi=200)


#
# # 生成 y=10*x1 + 3*x2 + 5 + noise 数据
x = np.array([[1, i, i] for i in range(50)])
y = np.array(50 * np.random.rand(50) + fn2(x, [5, 10, 3]))

print('gd')
gd = GD(x, y, fn2, lr=2e-5, verbose=True)
gd_theta,gd_theta_list = gd.step()
print('sgd')
sgd = SGD(x, y, fn2, lr=2e-5, verbose=True)
sgd_theta,sgd_theta_list = sgd.step()
print('mini_batch_sgd')
mini_batch_sgd = mini_batch_SGD(x, y, fn2, lr=2e-5, verbose=True, batch=4)
mini_batch_sgd_theta,mini_batch_sgd_theta_list = mini_batch_sgd.step()
print(gd_theta, sgd_theta, mini_batch_sgd_theta)

fig = plt.figure(3)
ax = Axes3D(fig)
ax.scatter(x[:, 1], x[:, 0], y,alpha=0.1)
plt.title('y=10*x1 + 3*x2 + 5 + noise')
ax.plot(x[:, 1], x[:, 0], fn2(x, gd_theta), color='black',alpha=0.7)
ax.plot(x[:, 1], x[:, 0], fn2(x, sgd_theta), color='yellow',alpha=0.3)
ax.plot(x[:, 1], x[:, 0], fn2(x, mini_batch_sgd_theta), color='deeppink',alpha=0.7)
ax.legend(['gb', 'sgd', 'mini_batch_sgd'])
plt.savefig('gd2.png', dpi=200)

fig = plt.figure(4)
ax = Axes3D(fig)
ax.plot(gd_theta_list[:, 0], gd_theta_list[:, 1], gd_theta_list[:,2], color='black')
plt.title('gb')
plt.savefig('result_Gradient_Descent/gd2_theta.png', dpi=200)
fig = plt.figure(5)
ax = Axes3D(fig)
ax.plot(sgd_theta_list[:, 0], sgd_theta_list[:, 1], sgd_theta_list[:,2], color='yellow')
plt.title('sgb')
plt.savefig('sgd2_theta.png', dpi=200)

fig = plt.figure(5)
ax = Axes3D(fig)
ax.plot(mini_batch_sgd_theta_list[:,0],mini_batch_sgd_theta_list[:,1],mini_batch_sgd_theta_list[:,2], color='deeppink')
plt.title('mini_batch_sgb')
plt.savefig('mini_batch_sgd2_theta.png', dpi=200)
