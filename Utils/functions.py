#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/4/6 19:42
# Tool :PyCharm

import scipy.io
import matplotlib.pyplot as plt
import numpy as np


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


def load_mat_data(data_path):
    """
        读取.mat文件数据
    :param data_path: 文件路径
    :return: fea:特征  label:标签
    """
    data = scipy.io.loadmat(data_path)
    fea = data['fea']
    label = data['gnd']
    return fea, label


if __name__ == '__main__':
    X = np.array([[2, 2], [-2, -1], [2, 2], [-2, -1]])
    y = np.array([1, 1, 1, 2])
    print(class_means(X, y))
