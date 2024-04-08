import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

minst = tf.keras.datasets.mnist  # load dataset from keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Normalization
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Creating Model
model = tf.keras.models.Sequential()  # basic sequential neural network

# create input layer. flatten makes 28 * 28 into 784 individual neurons
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# creating our hidden layers. This time we use relu instead of sigmoid for activation function
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# creating output layer. 10 neurons corresponding to output. This output uses softmax so value of all neurons equals 1
model.add(tf.keras.layers.Dense(128, activation='softmax'))

# compiling the model, choosing optimizer and loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training the model
model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)

print(f"Accuracy: {accuracy}%")
print(f"loss: {loss}")
