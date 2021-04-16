#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/4/16 11:43
# Tool :PyCharm
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

if __name__ == "__main__":
    pass
