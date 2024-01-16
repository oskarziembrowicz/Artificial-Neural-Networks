import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense

# Reading data
df = pd.read_csv('data_for_autoencoder.csv')
data = df.values

# Data normalization
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Splitting testing and training sets
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

input_dimensions = X_train.shape[1]

encoder_neurons = 64 # Number of neurons in the encoders first layer
inside_neurons = 32 # Number of newurons in the encoders inside layer

# Building the encoder
input_layer = Input(shape=(input_dimensions,))
# Encoder
encoder = Dense(encoder_neurons, activation="relu")(input_layer)
encoder = Dense(inside_neurons, activation="relu")(encoder)
#Decoder
decoder = Dense(encoder_neurons, activation="relu")(encoder)
decoder = Dense(input_dimensions, activation="sigmoid")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# Training the model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))