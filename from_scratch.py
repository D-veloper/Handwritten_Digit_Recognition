import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random

# Function to get the mnist training data set
def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

# Function to transform neuron values via sigmoid squishification
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

images, labels = get_mnist()
# 784 inputs each corresponding to greyscale value of pixel in 28 * 28 image
input_weights = np.random.uniform(-0.5, 0.5, (20, 784))  # the hidden layer has 20 neurons
# 10 outputs corresponding to numbers from 0 to 9.
output_weights = np.random.uniform(-0.5, 0.5, (10, 20))
# initially there is no bias
input_bias = np.zeros((20, 1))
output_bias = np.zeros((10, 1))

learning_rate = 0.01  # hyperparameter to determine step size during gradient descent
epochs = 3  # hyperparameter to determine how many times model will train on the entire data set
correct = 0  # variable to measure model accuracy later. not needed for training

for epoch in range(epochs):
    for img, lbl in zip(images, labels):
        # transform image and labels from vectors to matrices for matrix operations later
        img.shape += (1,)
        lbl.shape += (1,)

        # forward propagation from input to hidden layer
        h_pre = input_weights @ img + input_bias # calculate neuron values pre activation to pass into sigmoid func
        h = sigmoid(h_pre)

        # forward propagation from hidden to output layer
        o_pre = output_weights @ h + output_bias
        o = sigmoid(o_pre)

        # Cost / Mean Squared Error
        error = 1 / len(o) * np.sum((o - lbl) ** 2, axis=0)
        correct += int(np.argmax(o) == np.argmax(lbl))  # it's correct if prediction matches target/label

        # back propagation from output to hidden layer (cost function derivative)
        d_o = o - lbl
        output_weights += -learning_rate * d_o @ np.transpose(h)
        output_bias += -learning_rate * d_o

        # back propagation from hidden to input layer (activation function derivative)
        d_h = np.transpose(output_weights) @ d_o * (h * (1 - h))
        input_weights += -learning_rate * d_h @ np.transpose(img)
        input_bias += -learning_rate * d_h

    # Display accuracy for the current epoch and reset
    print(f"Accuracy: {round((correct/images.shape[0]) * 100, 2)}%")
    correct = 0

while True:
    index = random.randrange(0, 60000)
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = input_weights @ img.reshape(784, 1) + input_bias
    h = sigmoid(h_pre)
    # Forward propagation hidden -> output
    o_pre = output_weights @ h + output_bias
    o = sigmoid(o_pre)

    plt.title(f"The number is {o.argmax()}")
    plt.show()
    plt.pause(2)
    plt.close()
