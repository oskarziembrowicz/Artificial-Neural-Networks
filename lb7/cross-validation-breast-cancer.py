from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd

# Collecting data
breast_cancer_dataset = fetch_ucirepo(id=17)

X = breast_cancer_dataset.data.features
y = breast_cancer_dataset.data.targets

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)
# Data normalisation
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(breast_cancer_dataset.metadata)
# print(breast_cancer_dataset.variables)
# 32 rows x 7 columns

def create_model():
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(10, activation='softmax')
        tf.keras.layers.Dense(16, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# model1 = create_model()
# model1.fit(x_train, y_train, epochs=5)
# model1.evaluate(x_test, y_test)

