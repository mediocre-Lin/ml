#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/4/6 19:42
# Tool :PyCharm

import scipy.io
import matplotlib.pyplot as plt
import numpy as np





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
    pass