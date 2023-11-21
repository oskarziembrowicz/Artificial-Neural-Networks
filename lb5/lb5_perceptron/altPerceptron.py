import numpy as np

class Perceptron(object):
    def __init__(self, learning_parameter=0.01, number_of_iterations=50, random_seed=1):
        self.learning_parameter = learning_parameter
        self.number_of_iterations = number_of_iterations
        self.random_seed = random_seed
        self.number_of_errors = []

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_seed)
        self.wages = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.saved_weights = np.zeros((self.number_of_iterations + 1, len(self.wages)))
        self.saved_weights[0, :] = self.wages

        for i in range(self.number_of_iterations):
            errors = 0
            for xi, target in zip(X, y):
                # print(xi)
                update = self.learning_parameter * (target - self.predict_function(xi))
                self.wages[1:] += update * xi
                self.wages[0] += update * 1
                # print(self.wages[0:])
                errors += int(update != 0.0)
            self.number_of_errors.append(errors)
            self.saved_weights[i + 1, :] = self.wages
        # for i in range(0,4):
        # print(self.predict_function(X[i]))
        return self

    def fit_manual(self, X, y, w):
        self.wages = w

        for _ in range(self.number_of_iterations):
            errors = 0
            for xi, target in zip(X, y):
                # print(xi)
                update = self.learning_parameter * (target - self.predict_function(xi))
                self.wages[1:] += update * xi
                self.wages[0] += update * 1
                # (self.wages[0:])
                errors += int(update != 0.0)
            self.number_of_errors.append(errors)
        return self

    def net_input(self, X):
        # print(np.dot(X, self.wages[1:]) + self.wages[0])
        return np.dot(X, self.wages[1:]) + self.wages[0]

    def predict_function(self, X):
        # print(np.where(self.net_input(X) >= 0.0, 1, -1))
        return np.where(self.net_input(X) >= 0.0, 1, -1)

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

perceptron = Perceptron(learning_parameter=0.01, number_of_iterations=1000)

perceptron.fit(X_train, y_train)

predictions = perceptron.predict_function(X_test)

# print(perceptron.wages)
print(perceptron.saved_weights)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

plt.plot(range(0, perceptron.saved_weights.shape[0]), perceptron.saved_weights[:,0], label="w0", color="red")
plt.plot(range(0, perceptron.saved_weights.shape[0]), perceptron.saved_weights[:,1], label="w1", color="green")
plt.plot(range(0, perceptron.saved_weights.shape[0]), perceptron.saved_weights[:,2], label="w2", color="blue")
plt.plot(range(0, perceptron.saved_weights.shape[0]), perceptron.saved_weights[:,3], label="w3", color="yellow")
plt.xlabel("Changes")
plt.ylabel("Weights")
plt.legend(loc="lower left")
plt.show()