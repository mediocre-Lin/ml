# -*- encoding: utf-8 -*-
"""
@Time    : 2021/4/15 0:23
@Author  : kesth
@Software: PyCharm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from Optim.tree import Decison_Tree

demo_data = pd.read_csv('../../Data/demo2.csv', encoding='gbk')
x = demo_data.iloc[:,:-1].values

y = demo_data.iloc[:,-1].values

le = LabelEncoder()
for i in range(x.shape[1]):
    if i != 0:
        x[:,i] = le.fit_transform(x[:,i])
print(x)
print(y)
dt = Decison_Tree("CART",task_type='regression')
res = dt.fit(x,y)
