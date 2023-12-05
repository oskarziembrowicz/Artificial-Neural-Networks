import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np

# Collecting data:
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Coss-validation implementation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
for train, test in kfold.split(x_train, y_train):
    model = create_model()
    print(f'Training for fold {fold_no} ...')

    model.fit(x_train[train], y_train[train], epochs=5)

    scores = model.evaluate(x_train[test], y_train[test], verbose=0)
    print(f'Score for fold {fold_no}: '
          f'{model.metrics_names[0]} of {scores[0]}; '
          f'{model.metrics_names[1]} of {scores[1]*100}%')
    fold_no += 1

final_model = create_model()

final_model.fit(x_train, y_train, epochs=5)
final_model.evaluate(x_test, y_test)