# -*- coding: utf-8 -*-
''''''
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import os
import csv
from pprint import pprint


# 切断冪関数のh0~h3まで(前半)
def first_function(d, x):
    return x ** d


# 切断冪のh4~h7(後半)
def second_function(i, x, K):
    ci = 10 * i / (K + 1)
    return abs((x - ci) ** 3)


# f~(x)
def regression_spline(beta_hat, K, x):
    y = 0
    for i in range(4):
        y += beta_hat[i, 0] * first_function(i, x)
    for j in range(1, K + 1):
        y += beta_hat[3 + j, 0] * second_function(j, x, K)
    return y


def regression(K):
    # K = 4
    csv_path = 'TrainingDataForAssingment5.csv'
    data_num = 120
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        data_list = [row for row in reader]
    data_list.pop(0)

    for data in data_list:
        plt.plot(float(data[1]), float(data[2]), marker='.')
    # plt.savefig('data.png')

    y_list = [float(data[2]) for data in data_list]
    y_np = np.array(y_list)
    x_np = [float(data[1]) for data in data_list]
    # print(y_list)

    x_list = []
    for i in range(data_num):
        x = float(data_list[i][1])
        # print(x)
        sub_first_list = [first_function(d, x) for d in range(4)]
        sub_second_list = [second_function(i, x, K) for i in range(1, K + 1)]
        sub_list = sub_first_list + sub_second_list
        x_list.append(sub_list)

    x_mat = np.matrix(x_list)
    y_mat = np.matrix(y_list)
    # (xtx)^-1xty
    beta_hat = (x_mat.T * x_mat) ** -1 * x_mat.T * y_mat.T
    # print(beta_hat)

    x = np.linspace(0, 10, 100)
    y = np.array([regression_spline(beta_hat, K, p) for p in x])
    plt.plot(x, y)
    plt.savefig('k_' + str(K) + '.png')
    plt.clf()

    # CVLOOを求める(magic formula)
    h_for_cv_loo = x_mat * (x_mat.T * x_mat) ** -1 * x_mat.T

    cv_loo_magic = 0
    for i in range(data_num):
        cv_loo_magic += ((y_np[i] - regression_spline(beta_hat, K, x_np[i])) \
                / (1 - h_for_cv_loo[i, i])) ** 2
        
    cv_loo_magic = cv_loo_magic / data_num
    # print(cv_loo_magic)

    # CVLOOを求める(without magic formula)
    cv_loo_all = 0
    for i in range(data_num):
        cv_loo = 0
        for j in range(data_num):
            if i != j:
                cv_loo += (y_np[j] - regression_spline(beta_hat, K, x_np[j])) ** 2
        cv_loo_all += cv_loo / (data_num - 1)

    cv_loo_all = cv_loo_all/data_num
    print(cv_loo_all)

    return cv_loo_magic, cv_loo_all


def main():
    min_reg = 100
    for i in range(1, 16):
        reg_magic, reg = regression(i)
        print('i=' + str(i))
        print(reg_magic)
        print(reg)
        if reg < min_reg:
            min_reg = reg
            min_idx = i
    
    print('min K and min CVLOO')
    print(min_idx)
    print(min_reg)


if __name__ == "__main__":
    # regression(4)
    main()
