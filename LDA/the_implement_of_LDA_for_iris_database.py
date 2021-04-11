# -*- encoding: utf-8 -*-
"""
@Time    : 2021/4/11 11:30
@Author  : kesth
@Software: PyCharm
"""
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from Utils import LDA

iris_data = pd.read_csv('../Data/iris.csv')
x = iris_data.iloc[:,1:-1].values
y = iris_data.iloc[:,-1].values
le = LabelEncoder()
y = le.fit_transform(y)

lda = LDA()
x_lda = lda.fit_transform(x,y,n_component=2)

print(x_lda.shape)
