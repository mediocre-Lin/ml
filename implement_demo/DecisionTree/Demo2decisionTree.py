# -*- encoding: utf-8 -*-
"""
@Time    : 2021/4/15 0:23
@Author  : kesth
@Software: PyCharm
"""
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
    if isinstance(res,dict):
        if len(res['fea']):
            print(res)
    else:
        for i in range(len(res)):
            tree_print(res[i])
demo_data = pd.read_csv('../../Data/demo2.csv', encoding='gbk')
x = demo_data.iloc[:,:-1].values

y = demo_data.iloc[:,-1].values

le = LabelEncoder()
for i in range(x.shape[1]):
    if i != 0:
        x[:,i] = le.fit_transform(x[:,i])
dt = Decison_Tree("CART",task_type='regression')
res = dt.fit(x,y)
print(x)
print(y)
tree_print(res)
# dtc = DecisionTreeRegressor(splitter='best')
# dtc.fit(x,y)
# import pydotplus
# dot_data = tree.export_graphviz(dtc, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("demo.pdf")
