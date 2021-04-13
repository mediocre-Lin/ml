#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/3/30 13:02
# Tool :PyCharm
import random
import numpy as np
from Utils import Mean_Square_Error
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)


def class_means(X, y):
    """
        计算各类的均值

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means


def _conv(X):
    num = X.shape[0]
    x = X - np.nanmean(X, axis=0)
    return 1 / num * np.matmul(x.T, x)


def class_cov(X, y):
    """
      计算各类别的协方差矩阵
    :param X: x变量
    :param Y: y变量
    :return: 协方差矩阵
    """
    classes = np.unique(y)
    cov = np.zeros(shape=(len(classes), X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        X_g = X[y == group, :]
        cov[idx] = _conv(X_g)
    return cov


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


def _cal_ent(y):
    c_k = np.bincount(y)
    p_k = c_k / len(y)
    p_k = p_k[p_k!=0]
    return -sum(p_k * np.log2(p_k))


def _cal_ent_v(x, y):
    attr_x = np.unique(x)
    label = np.unique(y)
    ent_attr = 0.0

    for attr in attr_x:
        num_attr = sum(x == attr)
        for l in label:
            d_attr = x[(x == attr) & (y == l)]
            p_attr = len(d_attr) / num_attr
            ent_attr += 0 if p_attr == 0 else (num_attr / len(y)) * p_attr * (np.log2(len(d_attr) / num_attr))
    return ent_attr


def info_gain(x, y):
    Ent_D = _cal_ent(y)
    gain_feas = [Ent_D + _cal_ent_v(x[:, fea], y) for fea in range(x.shape[1])]
    return np.array(gain_feas)

def cal_purity(x):
    count = np.bincount(x)
    return np.max(count)/sum(count)


def split(x, y, target,note='-->'):
    tar_x = x[:,target]
    branch_fea = np.unique(tar_x)
    new_x = []
    for i, fea in enumerate(branch_fea):
            new_x.append( {'fea':np.array(x[tar_x == fea, :]),'label':y[tar_x==fea],'purity':cal_purity(y[tar_x==fea]) ,'note':note+'fea_'+str(target)+' = '+str(fea)+'-->'})
    return new_x


def id_3(data, max_info_gain_list):
    for idx, data_i in enumerate(data):
        if  data_i['purity'] != 1 or sum(max_info_gain_list)==0:
            info_gains = info_gain(data_i['fea'], data_i['label']) * max_info_gain_list
            max_info_gain = np.argmax(info_gains)
            max_info_gain_list[max_info_gain] = False
            data[idx] = split(data_i['fea'], data_i['label'], max_info_gain,data_i['note'])
            id_3(data[idx],max_info_gain_list)
    return data

class Decison_Tree(object):
    """
        决策树(Decision Tree)

    """

    def __init__(self, decision_type='id3'):
        self.type = decision_type
        self.result = None

    def fit(self, x, y):
        if self.type == 'id3':
            max_info_gain_list=[True for i in range(x.shape[1])]
            info_gains = info_gain(x, y)
            max_info_gain = np.argmax(info_gains)
            max_info_gain_list[max_info_gain]=False
            self.result = split(x,y,max_info_gain)
            self.result = id_3(self.result,max_info_gain_list)
            return self.result

    def predict(self):
        pass

    def print(self):
        pass


if __name__ == '__main__':
    print(sum([False,True]))
