# -*- encoding: utf-8 -*-
"""
@Time    : 2021/4/11 11:30
@Author  : kesth
@Software: PyCharm
"""
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from Optim.lda import LDA

iris_data = pd.read_csv('../../Data/iris.csv')
x = iris_data.iloc[:,1:-1].values
y = iris_data.iloc[:,-1].values
le = LabelEncoder()
y = le.fit_transform(y)

lda = LDA()
x_lda = lda.fit_transform(x,y,n_component=1)

plt.figure(1)
for label in [0,1,2]:
    x_l = x_lda[y==label,:]
    plt.scatter(x_l[:,0],x_l[:,0])
plt.show()
#
# fig = plt.figure(2)
# ax = Axes3D(fig)
# for label in [0,1,2]:
#     x_l = x_lda[y==label,:]
#     ax.scatter(x_l[:,0],x_l[:,1],x_l[:,2])
# plt.show()


# sk_lda = LinearDiscriminantAnalysis(n_components=1)
# sk_lda_x  = sk_lda.fit_transform(x,y)
# plt.figure(3)
# for label in [0,1,2]:
#     x_l = sk_lda_x[y==label,:]
#     plt.scatter(x_l[:,0],x_l[:,0])
# plt.show()

# fig = plt.figure(4)
# ax = Axes3D(fig)
# for label in [0,1,2]:
#     x_l = sk_lda_x[y==label,:]
#     ax.scatter(x_l[:,0],x_l[:,1],x_l[:,2])
# plt.show()