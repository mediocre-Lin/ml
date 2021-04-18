#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/4/18 13:14
# Tool :PyCharm

import os

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from Optim.tree import Decison_Tree


def tree_print(res):
    if isinstance(res, dict):
        if len(res['fea']):
            print(res)
    else:
        for i in range(len(res)):
            tree_print(res[i])


demo_data = pd.read_csv('../../Data/西瓜数据集3.0.csv', encoding='gbk')
x = demo_data.iloc[:,:-3].values

y = demo_data.iloc[:,-1].values

le = LabelEncoder()
for i in range(x.shape[1]):
        x[:,i] = le.fit_transform(x[:,i])
y = le.fit_transform(y)
print(x)
print(y)
dt = Decison_Tree("id3")
res = dt.fit(x,y)

tree_print(res)
