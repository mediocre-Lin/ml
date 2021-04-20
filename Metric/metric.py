#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/4/18 16:22
# Tool :PyCharm
import numpy as np


def Accuray(pre, tar):
    return (1 - (pre != tar)) / len(tar)
