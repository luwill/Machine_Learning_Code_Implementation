import numpy as np


class Perceptron:
    def __init__(self):
        pass

    def initialize_with_zeros(self, dim):
        w = np.zeros(dim)
        b = 0.0
        return w, b

    def sign(self, x, w, b):
        return np.dot(x, w) + b

    def train(self, X_train, y_train, learning_rate):
        w, b = self.initialize_with_zeros(X_train.shape[1])
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]

                if y * self.sign(X, w, b) <= 0:
                    w = w + learning_rate * np.dot(y, X)
                    b = b + learning_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True

            params = {
                'w': w,
                'b': b
            }
        return params
