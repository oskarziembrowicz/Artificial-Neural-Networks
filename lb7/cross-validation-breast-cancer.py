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
        tf.keras.layers.Flatten(input_shape=(32, 7)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        # tf.keras.layers.Dense(16, activation='relu', input_shape=(x_train.shape[1],)),
        # tf.keras.layers.Dense(1, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

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

# model1 = create_model()
# model1.fit(x_train, y_train, epochs=5)
# model1.evaluate(x_test, y_test)

