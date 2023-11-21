import numpy as np


class Perceptron(object):
    def __init__(self, learning_parameter=0.01, number_of_iterations=50, random_seed=1):
        self.learning_parameter = learning_parameter
        self.number_of_iterations = number_of_iterations
        self.random_seed = random_seed
        self.number_of_errors = []

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_seed)
        self.wages = rgen.normal(loc=0.0, scale=0.01 + X.shape[1])

        for _ in range(self.number_of_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_parameter * (target - self.predict_function(xi))
                self.wages[1:] += update * xi
                self.wages[0] += update * 1
                errors += int(update != 0.0)
            self.number_of_errors.append(errors)
        return self

    def fit_manual(self, X, y, w):
        self.wages = w

        for _ in range(self.number_of_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_parameter * (target - self.predict_function(xi))
                self.wages[1:] += update * xi
                self.wages[0] += update * 1
                errors += int(update != 0.0)
            self.number_of_errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.wages[1:]) + self.wages[0] * 1

    def predict_function(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


perceptron = Perceptron()
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = [1, 0, 1, 0]
# w = [1, 1, 1]
perceptron.fit(X, y)
# perceptron.fit_manual(X, y, w)