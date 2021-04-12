# -*- encoding: utf-8 -*-
"""
@Time    : 2021/4/12 10:12
@Author  : kesth
@Software: PyCharm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from Utils import Decison_Tree

demo_data = pd.read_excel('../Data/demo_decision_Tree.xlsx')
x = demo_data.iloc[:,:-1].values
y = demo_data.iloc[:,-1].values

le = LabelEncoder()
for i in range(x.shape[1]):
    x[:,i] = le.fit_transform(x[:,i])
y = le.fit_transform(y)
all = pd.DataFrame(np.hstack((x,y.reshape(-1,1))),columns=['年龄','收入','学生','信用','买了电脑'])
print(all)
dt = Decison_Tree()
res = dt.fit(x,y)
for i in range(len(res)):
    print(res[i])
    print()

