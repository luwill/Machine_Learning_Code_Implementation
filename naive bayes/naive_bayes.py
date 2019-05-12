#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

'''
@author: louwill
@contact: ygnjd2016@gmail.com
@file: naive_bayes.py
@time: 2019/5/12 9:41
'''

import numpy as  np
import pandas as pd

class Naive_Bayes:
    def __init__(self):
        pass

    # 朴素贝叶斯训练过程
    def nb_fit(self, X, y):
        classes = y[y.columns[0]].unique()
        class_count = y[y.columns[0]].value_counts()
        # 类先验概率
        class_prior = class_count / len(y)
        # 计算类条件概率
        prior = dict()
        for col in X.columns:
            for j in classes:
                p_x_y = X[(y == j).values][col].value_counts()
                for i in p_x_y.index:
                    prior[(col, i, j)] = p_x_y[i] / class_count[j]

        return classes, class_prior, prior

    # 预测新的实例
    def predict(self, X_test):
        res = []
        for c in classes:
            p_y = class_prior[c]
            p_x_y = 1
            for i in X_test.items():
                p_x_y *= prior[tuple(list(i) + [c])]
            res.append(p_y * p_x_y)
        return classes[np.argmax(res)]


if __name__ == "__main__":
    x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    x2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    X = df[['x1', 'x2']]
    y = df[['y']]
    X_test = {'x1': 2, 'x2': 'S'}

    nb = Naive_Bayes()
    classes, class_prior, prior = nb.nb_fit(X, y)
    print('测试数据预测类别为：', nb.predict(X_test))

