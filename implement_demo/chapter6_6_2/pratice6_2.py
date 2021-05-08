#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/5/8 16:58
# Tool :PyCharm
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def plot_svc(feas,pre,sv=None):
    plt.scatter(feas[pre==0,0],feas[pre==0,1])
    plt.scatter(feas[pre==1,0],feas[pre==1,1])
    if sv is None:
        plt.legend(['Y', 'N'])
    else:
        plt.scatter(feas[sv,0],feas[sv,1],marker='o',c='',edgecolors='r')
        plt.legend(['Y','N','SuportVector'])


data = pd.read_csv("../../Data/西瓜数据集3.0.csv", encoding='gbk')
feas = data.iloc[:,-3:-1].values
label = data.iloc[:,-1].values
le = LabelEncoder()
label = le.fit_transform(label)
svm_linear = SVC(kernel='linear')
svm_rbf = SVC(kernel='rbf')
svm_linear.fit(feas,label)
svm_rbf.fit(feas,label)
print(svm_linear.support_)
print(svm_rbf.support_)
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plot_svc(feas,label)
plt.title('Origin Data')
plt.subplot(1,3,2)
plot_svc(feas,svm_linear.predict(feas),svm_linear.support_)
plt.title('Linear SVM')
plt.subplot(1,3,3)
plot_svc(feas,svm_rbf.predict(feas),svm_rbf.support_)
plt.title('RBF SVM')
plt.savefig('pratice6_2.png',dpi=100)

plt.show()

