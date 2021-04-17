#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Kesth
# createtime: 2021/4/16 11:42
# Tool :PyCharm

import numpy as np


def _cal_ent(y):
    c_k = np.bincount(y)
    p_k = c_k / len(y)
    p_k = p_k[p_k != 0]
    return -sum(p_k * np.log2(p_k))


def _cal_ent_v(x, y):
    attr_x = np.unique(x)
    label = np.unique(y)
    ent_attr = 0.0

    for attr in attr_x:
        num_attr = sum(x == attr)
        for l in label:
            d_attr = x[(x == attr) & (y == l)]
            p_attr = len(d_attr) / num_attr
            ent_attr += 0 if p_attr == 0 else (num_attr / len(y)) * p_attr * (np.log2(len(d_attr) / num_attr))
    return ent_attr


def info_gain(x, y):
    Ent_D = _cal_ent(y)
    gain_feas = [Ent_D + _cal_ent_v(x[:, fea], y) for fea in range(x.shape[1])]
    return np.array(gain_feas)


def cal_purity(x):
    if len(x) == 0:
        return 0
    else:
        count = np.bincount(x)
        return np.max(count) / sum(count)


def _iv(x, y):
    attr_x = np.unique(x)
    IV = 0.0

    for attr in attr_x:
        num_attr = sum(x == attr)
        p_attr = (num_attr / len(y))
        IV += 0.0 if p_attr == 0 else p_attr * (np.log2(p_attr))
    return IV


def gain_ratio(x, y):
    Ent_D = _cal_ent(y)
    gain_rat = [(Ent_D + _cal_ent_v(x[:, fea], y) / _iv(x[:, fea], y)) for fea in range(x.shape[1])]
    return np.array(gain_rat)


# CART

def _cal_gini(x, y):
    label = np.unique(y)
    gini = 0.0
    for k in label:
        p_k = len(x[y == k]) / len(y)
        gini += p_k ** 2
    return 1 - gini


def _cal_index(x, y, attr, continuous_feas, type='classifier'):
    if continuous_feas:
        x_selected = x <= attr
    else:
        x_selected = x == attr
    if type == 'classifier':
        index = len(x[x_selected]) / len(y) * _cal_gini(x[x_selected], y[x_selected]) + len(x[~x_selected]) / len(
            y) * _cal_gini(x[~x_selected], y[~x_selected])
    else:
        index = _reg_conv(y[x_selected]) + _reg_conv(y[~x_selected])
    return index


def cal_split_index(x, y, attr_martix, continuous_feas, tree_type='classifier'):
    gini_index_martix = np.zeros_like(attr_martix) + 1e3
    for row in range(attr_martix.shape[0]):
        fea = x[:, row]
        for col in range(attr_martix.shape[1]):
            if attr_martix[row, col] != -1:
                attr = attr_martix[row, col] if continuous_feas[row] else col
                gini_index_martix[row, col] = _cal_index(fea, y, attr, continuous_feas[row], type=tree_type)
    return gini_index_martix


def _reg_conv(y):
    return np.sqrt(sum(y ** 2) - len(y) * np.mean(y) ** 2)


def init_all_fea_attr(x, numeric_fea):
    attr_num = []
    for fea in range(len(numeric_fea)):
        if numeric_fea[fea]:
            attr_num.append(len(np.unique(x[:, fea])) - 1)
        else:
            attr_num.append(len(np.unique(x[:, fea])))
    attr_martix = np.zeros((x.shape[1], max(attr_num))) - 1
    for idx, num in enumerate(attr_num):
        if numeric_fea[idx]:
            attr_martix[idx, :] = continuous2dispersed(x[:, idx])
        else:
            if num == 2:
                attr_martix[idx, 0] = 1
            else:
                attr_martix[idx, :num] = 1
    return attr_martix


def adjust_att_martix(attr_matrix):
    for i in range(attr_matrix.shape[0]):
        if sum(-1 != attr_matrix[i, :]) == 2:
            attr_matrix[i, np.argmax(attr_matrix[i, :])] = -1
    return attr_matrix


def continuous_fea_check(x):
    fea_max = max(x)
    if int(fea_max) != fea_max:
        return True
    else:
        if list(range(int(fea_max) + 1)) != list(np.unique(x)):
            return True
    return False


def num2coordinate(num, shape):
    return [int(num / shape[1]), num % shape[1]]


def continuous2dispersed(num_fea):
    num_fea_list = np.unique(num_fea)

    split_list = 0.5 * (num_fea_list[:-1] + num_fea_list[1:])
    return split_list


def split(x, y, attrs_martix, target='None', note='-->', type='gini', mini_gini=None, continuous_fea=False):
    if type == 'gini':
        tar_x = x[:, mini_gini[0]]
        if continuous_fea:
            new_x = []
            val = attrs_martix[mini_gini[0], mini_gini[1]]
            seleted = tar_x <= val
            new_x.append(
                {'fea': np.array(x[seleted]), 'label': y[seleted],
                 'purity': cal_purity(y[seleted]),
                 'note': note + 'fea_' + str(mini_gini[0]) + ' <= ' + str(val) + '-->'})
            new_x.append(
                {'fea': np.array(x[~seleted]), 'label': y[~seleted],
                 'purity': cal_purity(y[~seleted]),
                 'note': note + 'fea_' + str(mini_gini[0]) + '> ' + str(val) + '-->'})
        else:
            new_x = []
            new_x.append(
                {'fea': np.array(x[tar_x == mini_gini[1], :]), 'label': y[tar_x == mini_gini[1]],
                 'purity': cal_purity(y[tar_x == mini_gini[1]]),
                 'note': note + 'fea_' + str(mini_gini[0]) + ' = ' + str(mini_gini[1]) + '-->'})
            new_x.append(
                {'fea': np.array(x[tar_x != mini_gini[1], :]), 'label': y[tar_x != mini_gini[1]],
                 'purity': cal_purity(y[tar_x != mini_gini[1]]),
                 'note': note + 'fea_' + str(mini_gini[0]) + ' != ' + str(mini_gini[1]) + '-->'})

    else:
        tar_x = x[:, target]
        branch_fea = np.unique(tar_x)
        new_x = []
        for i, fea in enumerate(branch_fea):
            new_x.append(
                {'fea': np.array(x[tar_x == fea, :]), 'label': y[tar_x == fea], 'purity': cal_purity(y[tar_x == fea]),
                 'note': note + 'fea_' + str(target) + ' = ' + str(fea) + '-->'})

    return new_x


def id_3(data, max_info_gains_list):
    for idx, data_i in enumerate(data):
        max_info_gain_list = max_info_gains_list
        if data_i['purity'] != 1 or sum(max_info_gain_list) == 0:
            info_gains = info_gain(data_i['fea'], data_i['label']) * max_info_gain_list
            max_info_gain = np.argmax(info_gains)
            max_info_gain_list[max_info_gain] = False
            data[idx] = split(data_i['fea'], data_i['label'], max_info_gain, data_i['note'])
            id_3(data[idx], max_info_gain_list)
    return data


def C4_5(data, max_gain_ratios_list):
    for idx, data_i in enumerate(data):
        max_gain_ratio_list = max_gain_ratios_list
        if data_i['purity'] != 1 or sum(max_gain_ratio_list) == 0:
            info_gains = info_gain(data_i['fea'], data_i['label']) * max_gain_ratio_list
            gain_ratios = gain_ratio(data_i['fea'], data_i['label']) * info_gains > np.mean(info_gains)
            max_gain_ratio = np.argmax(gain_ratios)
            max_gain_ratio_list[max_gain_ratio] = False
            data[idx] = split(data_i['fea'], data_i['label'], max_gain_ratio, data_i['note'])
            C4_5(data[idx], max_gain_ratio_list)
    return data


def CART(data, attrs_martix, numeric_fea, task_type):
    for idx, data_i in enumerate(data):
        attr_martix = attrs_martix
        if data_i['purity'] != 1 and np.sum(-1 != attr_martix) != 0:
            gini_martix = cal_split_index(data_i['fea'], data_i['label'], attr_martix, numeric_fea, task_type)
            min_gini = num2coordinate(np.argmin(gini_martix), gini_martix.shape)
            data[idx] = split(data_i['fea'], data_i['label'], type='gini', mini_gini=min_gini, attrs_martix=attr_martix,
                              continuous_fea=numeric_fea[min_gini[0]],note=data_i['note'])
            attr_martix[min_gini[0], min_gini[1]] = -1
            attr_martix = adjust_att_martix(attr_matrix=attr_martix)
            CART(data[idx], attr_martix, numeric_fea, task_type)
    return data


class Decison_Tree(object):
    """
        决策树(Decision Tree)

    """

    def __init__(self, decision_type='id3', task_type='classifier'):
        self.type = decision_type
        self.result = None
        self.numeric_fea = []
        self.task_type = task_type

    def fit(self, x, y):
        if self.type == 'id3':
            max_info_gain_list = [True for i in range(x.shape[1])]
            info_gains = info_gain(x, y)
            max_info_gain = np.argmax(info_gains)
            max_info_gain_list[max_info_gain] = False
            self.result = split(x, y, max_info_gain)
            self.result = id_3(self.result, max_info_gain_list)
            return self.result
        elif self.type == 'C4_5':
            max_gain_ratio_list = [True for i in range(x.shape[1])]
            info_gains = info_gain(x, y)
            gain_ratios = gain_ratio(x, y) * info_gains > np.mean(info_gains)
            max_gain_ratio = np.argmax(gain_ratios)
            max_gain_ratio_list[max_gain_ratio] = False
            self.result = split(x, y, max_gain_ratio)
            self.result = C4_5(self.result, max_gain_ratio_list)
            return self.result
        elif self.type == 'CART':
            self.numeric_fea = [continuous_fea_check(x[:, fea]) for fea in range(x.shape[1])]
            attr_martix = init_all_fea_attr(x, self.numeric_fea)
            gini_martix = cal_split_index(x, y, attr_martix, self.numeric_fea, self.task_type)
            min_gini = num2coordinate(np.argmin(gini_martix), gini_martix.shape)
            self.result = split(x, y, type='gini', mini_gini=min_gini, attrs_martix=attr_martix,
                                continuous_fea=self.numeric_fea[min_gini[0]])
            attr_martix[min_gini[0], min_gini[1]] = -1
            attr_martix = adjust_att_martix(attr_matrix=attr_martix)
            CART(self.result, attr_martix, self.numeric_fea, self.task_type)
            print(self.result)
            # return self.result

    def predict(self):
        pass

    def print(self):
        pass


if __name__ == "__main__":
    pass
