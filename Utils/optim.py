#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/3/30 13:02
# Tool :PyCharm
import random
import numpy as np
from Utils import Mean_Square_Error, class_means, class_cov
import matplotlib.pyplot as plt


class GD(object):
    """
      Gradient Descent
      梯度下降算法

    """

    def __init__(self, x, y, fn, cost_fn=Mean_Square_Error, lr=1e-3, max_iter=50000, epsilon=1e-7, verbose=True):
        self.x = x
        self.y = y
        self.fn = fn
        self.theta_nums = x.shape[1]
        self.sample_nums = x.shape[0]
        self.theta = np.zeros(self.theta_nums)  # np.random.random(self.theta_nums)
        self.grad = np.zeros(self.theta_nums)
        self.cost_fn = cost_fn
        self.max_iter = max_iter
        self.lr = lr
        self.epsilon = epsilon
        self.verbose = verbose

    # 计算全部数据的梯度
    def grad_cal(self):
        for theta_idx in range(self.theta_nums):
            if theta_idx == 0:
                self.grad[theta_idx] = np.sum(np.array(self.fn(self.x, self.theta)) - self.y) / self.sample_nums
            else:
                self.grad[theta_idx] = np.sum(
                    np.array((self.fn(self.x, self.theta) - self.y)) * self.x[:, theta_idx]) / self.sample_nums
        return self.grad

    # 计算损失
    def cost_cal(self):
        y_pre = self.fn(self.x, self.theta)
        return self.cost_fn(self.sample_nums, y_pre, self.y)

    # 迭代更新参数
    def step(self):
        previous_cost = 0
        iters = 0
        thera_list = np.array([self.theta])
        while iters < self.max_iter:
            iters += 1
            self.grad = self.grad_cal()
            self.theta -= self.lr * self.grad
            thera_list = np.concatenate((thera_list, [self.theta]), axis=0)
            cur_cost = self.cost_cal()
            if abs(cur_cost - previous_cost) < self.epsilon:
                break
            else:
                if self.verbose:
                    print("iter%5s : theta :%20s  cost :%10.10s epsilon:%10s" % (
                        iters, self.theta, cur_cost, abs(cur_cost - previous_cost)))
                previous_cost = cur_cost

        if self.verbose:
            print("finall :theta :%s " % (self.theta))
        return self.theta, thera_list


class SGD(GD):
    # 随机梯度下降
    # 随机取一个样本进行梯度计算

    def __init__(self, x, y, fn, cost_fn=Mean_Square_Error, lr=1e-3, max_iter=50000, epsilon=1e-7, verbose=True):
        super().__init__(x, y, fn, cost_fn, lr, max_iter, epsilon, verbose)
        self.idx = self.random_idx()

    def random_idx(self):
        return random.randint(0, self.sample_nums - 1)

    def grad_cal(self):
        self.idx = self.random_idx()
        for theta_idx in range(self.theta_nums):
            if theta_idx == 0:
                self.grad[theta_idx] = np.array(self.fn(self.x[self.idx, :].reshape(1, -1), self.theta)) - self.y[
                    self.idx]
            else:
                self.grad[theta_idx] = np.array(
                    (self.fn(self.x[self.idx, :].reshape(1, -1), self.theta) - self.y[self.idx])) * self.x[
                                           self.idx, theta_idx]
        return self.grad

    # 计算损失

    def cost_cal(self):
        y_pre = self.fn(self.x[self.idx, :].reshape(1, -1), self.theta)
        return self.cost_fn(1, y_pre, self.y[self.idx])


class mini_batch_SGD(SGD):
    # 小批量随机梯度下降
    # batch 为批量数
    def __init__(self, x, y, fn, cost_fn=Mean_Square_Error, lr=1e-3, max_iter=50000, epsilon=1e-7, verbose=True,
                 batch=2):
        self.batch = batch
        super(mini_batch_SGD, self).__init__(x, y, fn, cost_fn, lr, max_iter, epsilon, verbose)

    # 小批量随机梯度下降
    def grad_cal(self):
        self.idx = random.sample(range(self.sample_nums), self.batch)
        idx = self.idx
        for theta_idx in range(self.theta_nums):
            if theta_idx == 0:
                self.grad[theta_idx] = np.sum(np.array(self.fn(self.x[idx, :], self.theta)) - self.y[idx]) / self.batch
            else:
                self.grad[theta_idx] = np.sum(np.array((self.fn(self.x[idx, :], self.theta) - self.y[idx])) * self.x[
                    idx, theta_idx]) / self.batch
        return self.grad

    # 计算损失

    def cost_cal(self):
        y_pre = self.fn(self.x[self.idx, :], self.theta)
        return self.cost_fn(self.batch, y_pre, self.y[self.idx])


class LDA(object):
    """
        线性判别分析
        Linear Discriminant Analysis
    """

    def __init__(self):
        self.w = None

    def fit(self, X, y, n_component=None):

        X = np.array(X)
        y = np.array(y)
        classes = np.unique(y)
        num_classes = len(classes)
        n_components = num_classes if n_component is None else n_component

        mean_classes = class_means(X, y)
        mean_all = np.mean(mean_classes, axis=0)
        s_w = np.zeros((X.shape[1], X.shape[1]))
        for idx, c in enumerate(classes):
            x_c = X[y == c, :]
            s_w += np.dot((x_c - mean_classes[idx]).T, (x_c - mean_classes[idx]))

        s_b = np.zeros((mean_classes.shape[1], mean_classes.shape[1]))
        for idx, c in enumerate(classes):
            mean_c = mean_classes[idx]
            num_c = len(X[y == c, :])
            s_b += num_c * np.dot((mean_c - mean_all).T, (mean_c - mean_all))

        # 计算S_w^{-1} S_b的特征值和特征矩阵
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
        sorted_indices = np.argsort(eig_vals)
        # 提取前k个特征向量
        self.w = eig_vecs[:, sorted_indices[:-n_components - 1:-1]]
        return self

    def transform(self, x):
        return np.dot(x, self.w)

    def fit_transform(self, x, y, n_component=None):
        self.fit(x, y, n_component)
        return self.transform(x)


if __name__ == '__main__':
    print(np.zeros(3))
