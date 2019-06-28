#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 15:21
# @Author  : louwill
# @File    : lasso.py
# @mail: ygnjd2016@gmail.com


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Ridge():
    def __init__(self):
        pass
    
    def prepare_data(self):
        data = pd.read_csv('./abalone.csv')
        data['Sex'] = data['Sex'].map({'M': 0, 'F': 1, 'I': 2})
        X = data.drop(['Rings'], axis=1)
        y = data[['Rings']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
        return X_train, y_train, X_test, y_test
    
    def initialize(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b

    def l2_loss(self, X, y, w, b, alpha):
        num_train = X.shape[0]
        num_feature = X.shape[1]
        y_hat = np.dot(X, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train + alpha * (np.sum(np.square(w)))
        dw = np.dot(X.T, (y_hat - y)) / num_train + 2 * alpha * w
        db = np.sum((y_hat - y)) / num_train
        return y_hat, loss, dw, db

    def ridge_train(self, X, y, learning_rate=0.01, epochs=1000):
        loss_list = []
        w, b = self.initialize(X.shape[1])
        for i in range(1, epochs):
            y_hat, loss, dw, db = self.l2_loss(X, y, w, b, 0.1)
            w += -learning_rate * dw
            b += -learning_rate * db
            loss_list.append(loss)
        
            if i % 100 == 0:
                print('epoch %d loss %f' % (i, loss))
            params = {
                'w': w,
                'b': b
            }
            grads = {
                'dw': dw,
                'db': db
            }
        return loss, loss_list, params, grads
    
    
    def predict(self, X, params):
        w = params['w']
        b = params['b']
        y_pred = np.dot(X, w) + b
        return y_pred
    
  

if __name__ == '__main__':
    ridge = Ridge()
    X_train, y_train, X_test, y_test = ridge.prepare_data()
    loss, loss_list, params, grads = ridge.ridge_train(X_train, y_train, 0.01, 1000)
    print(params)

    
    