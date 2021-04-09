#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/3/30 13:02
# Tool :PyCharm

import numpy as np
import math


# 均方误差

def Mean_Square_Error(m,Y, Y_tru):
    return np.sum((np.array(Y) - np.array(Y_tru)) ** 2) / (2 * m)


if __name__ == '__main__':
    y = [2, 2]
    y_tru = [4, 4]
    print(Mean_Square_Error(y, y_tru))
