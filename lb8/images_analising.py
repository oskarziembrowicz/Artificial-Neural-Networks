import os
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img


# Path to images directory
data_dir = 'images/images'

# Setting up images and labels
images = []
labels = []

# print(len(os.listdir(data_dir)))

for file in os.listdir(data_dir):
    if file.endswith('.jpg') or file.endswith('.png'):
        img_path = os.path.join(data_dir, file)
        img = load_img(img_path, target_size=(150,150))
        img = img_to_array(img)
        images.append(img)
        if 'square' in file:
            labels.append(0)
        else:
            labels.append(1)

images = np.array(images)
labels = np.array(labels)

# Preparing training and testing data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

# Defining the CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=20, validation_data=(X_test, y_test))

# Evaluating the model
model.evaluate(X_test, y_test)