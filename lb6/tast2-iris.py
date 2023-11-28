from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Loading dataset
cancer = datasets.load_iris()
X = cancer.data
y = cancer.target

# Splitting data into testing and training data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)
# Data normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trainig the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluating
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Testing loss: {loss}, Testing accuracy: {accuracy}")

# Analysing
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy in time')
plt.ylabel('Accuracy')
plt.xlabel("Epoch")
plt.legend()
plt.show()