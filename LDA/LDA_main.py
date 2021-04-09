#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/4/6 20:33
# Tool :PyCharm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import numpy as np
import math
from Utils import load_mat_data
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def plot_LDA(converted_X,y):
    fig = plt.figure(2)
    ax = Axes3D(fig)
    for target in [0,1,2,3,4,5,6,7,8,9]:
        pos = (y==target).ravel()
        X=converted_X[pos,:]
        ax.scatter(X[:,0],X[:,1],X[:,2])
    plt.legend([0,1,2,3,4,5,6,7,8,9])
    fig.suptitle("MNIST After LDA")
    plt.savefig('lda1.png',dpi=500)
    plt.show()


    plt.figure(3)
    for target in [0,1,2,3,4,5,6,7,8,9]:
        pos = (y==target).ravel()
        X=converted_X[pos,:]
        plt.scatter(X[:,0],X[:,1])
    plt.legend([0,1,2,3,4,5,6,7,8,9])
    fig.suptitle("MNIST After LDA")
    plt.savefig('lda2.png',dpi=500)
    plt.show()

    plt.figure(4)
    for target in [0,1,2,3,4,5,6,7,8,9]:
        pos = (y==target).ravel()
        X=converted_X[pos,:]
        plt.subplot(2,5,target+1)
        plt.axis('off')
        plt.scatter(X[:,0],X[:,1])
        plt.title('category:'+str(target))
    plt.savefig('lda3.png',dpi=500)
    plt.show()

    plt.figure(5)
    plt.title("MNIST After LDA")
    for target in [0,1,2,3,4,5,6,7,8,9]:
        pos = (y==target).ravel()
        X=converted_X[pos,:]
        plt.hist(X[0],label=str(target),alpha=0.7,bins=30)
    plt.savefig('lda4.png',dpi=500)

    plt.show()




MNIST_path = '../Data/MNIST.mat'
MNIST_fea,MNIST_label = load_mat_data(MNIST_path)
MNIST_fea = np.array(MNIST_fea)
# MNIST_fea = (MNIST_fea - MNIST_fea.mean()) / MNIST_fea.std()
X_train,y_train,X_test,y_test = MNIST_fea[:60000],MNIST_label[:60000],MNIST_fea[60000:],MNIST_label[60000:]



X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# pca = PCA(n_components=392)
# pca.fit(MNIST_fea)
#
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)
# print(X_train.shape)
# print(X_test.shape)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train.ravel())

converted_X = np.dot(MNIST_fea,np.transpose(lda.coef_))+lda.intercept_
print(converted_X.shape)
plot_LDA(converted_X,MNIST_label)
print("Score: %.2f" %(lda.score(X_test,y_test.reshape(-1))))