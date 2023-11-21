import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, number_of_iterations=1000):
        #Inicjalizacja preceptronu z określonym współczynnikiem uczenia i liczba iteracji
        self.learning_rate = learning_rate  # Współczynnik uczenia
        self.number_of_iterations = number_of_iterations    # Liczba iteracji
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Trenowanie modelu na danych X(cechy) i y(etykiety)
        n_samples, n_features = X.shape

        # Inicjalizacja wag i biasu
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Pętla ucząca
        for _ in range(self.number_of_iterations):
            # for idx, x_i in enumerate(X):
            for x_i, target in zip(X, y):
                # Obliczenie wyjścia liniowego
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Zastosowanie funkcji aktywacji
                y_predicted = self.activation_function(linear_output)

                # Aktualizacja wag i biasu
                update = self.learning_rate * (target - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        # Przewidywanie etykier dla nowych danych
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    def activation_function(self, x):
        # Funkcja aktywacji: zwraca 1 jeśli x>=0, lub 0
        return np.where(x >= 0, 1, 0)


from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# X = np.array([[1, 2], [2, 3], [3, 4]])
# y = np.array([0, 1, 0])

# Trenowanie modelu
perceptron = Perceptron(learning_rate=0.01, number_of_iterations=1000)
perceptron.fit(X_train, y_train)

# print(perceptron.weights)

# Uzycie modelu
predictions = perceptron.predict(X_test)

print(perceptron.weights)
print(predictions)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")