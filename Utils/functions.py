#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/4/6 19:42
# Tool :PyCharm

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def load_mat_data(data_path):
    data = scipy.io.loadmat(data_path)
    fea = data['fea']
    label = data['gnd']
    return fea,label


if __name__ == '__main__':
    MNIST_data = '../Data/MNIST.mat'
    M_fea, M_label = load_mat_data(MNIST_data)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.axis('off')
        plt.imshow(np.array(M_fea[i]).reshape(-1,28),'gray')
    plt.show()
    USPS_data = '../Data/USPS.mat'
    USPS_fea, USPS_label = load_mat_data(USPS_data)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.axis('off')
        plt.imshow(np.array(USPS_fea[i]).reshape(-1,16),'gray')
    plt.show()

    COIL20 = '../Data/COIL20.mat'
    COIL20_fea, COIL20_label = load_mat_data(COIL20)
    for i in range(9):
        plt.subplot(3,9,i+1)
        plt.axis('off')
        plt.imshow(np.array(COIL20_fea[i]).reshape(-1,32),'gray')
        plt.subplot(3,9,i+10)
        plt.axis('off')
        plt.imshow(np.array(COIL20_fea[72+i]).reshape(-1,32),'gray')
        plt.subplot(3,9,i+19)
        plt.axis('off')
        plt.imshow(np.array(COIL20_fea[2*72+i]).reshape(-1,32),'gray')

    plt.show()