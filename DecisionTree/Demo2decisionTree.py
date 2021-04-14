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
from Utils import Decison_Tree

demo_data = pd.read_csv('../Data/demo_decision_Tree2.csv',encoding='utf-8')
print(demo_data)
x = demo_data.iloc[:,:-1].values

y = demo_data.iloc[:,-1].values

le = LabelEncoder()
for i in range(x.shape[1]):
    if i != 2:
        x[:,i] = le.fit_transform(x[:,i])
y = le.fit_transform(y)
dt = Decison_Tree("CART")
res = dt.fit(x[:,:-1],y)
for i in range(len(res)):
    print(res[i])
    print()