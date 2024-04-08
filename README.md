# Handwritten Digit Recognition

## Introduction

Handwritten Digit Recognition is like the "Hello World" of Machine Learning. It is a problem that is not trivial to solve but also not too difficult so its a great starting point. The goal of this project was to learn and master core concepts related to artificial neural network through building a neural network from scratch that could recognise handwritten digits. To accomplish all this, I worked through the book tittled "Neural Networks and Deep Learning" by Michael Nielsen and followed some other sources, all of which are credited below. Finally, understanding the core of ANNs made it easier to replicate the neural network using python libraries.

---

### Skills Cultivated

**Hard Skills:**
- **Neural Network Implementation**: Developing neural networks from scratch.
- **Python Programming**: Utilizing Python for implementation and experimentation.
- **Data Preprocessing**: Processing and preparing MNIST dataset for training.
- **Model Evaluation**: Assessing model performance using metrics like accuracy.
- **Gradient Descent Optimization**: Optimizing network parameters for improved performance.
- **Understanding Activation Functions**: Exploring the characteristics and applications of activation functions.
- **Image Processing**: Manipulating and processing grayscale images for input data.

**Soft Skills:**
- **Self-Directed Learning**: Actively seeking and acquiring knowledge for personal development.

---

### Libraries Used
- **NumPy**: Utilized for mathematical operations and array manipulation.
- **Matplotlib**: Visualizing images and network outputs for analysis.
- **TensorFlow & Keras**: Employed for constructing neural networks efficiently.

---

### Neural Network Architecture

- #### The Perceptron
  First, I got aqcuainted with the perceptron which takes several binary inputs and produces a single binary output. To produce the output, perceptrons have weights- real numbers that embody the significance of respective inputs to outputs. And a neuron's output is 0 or 1 based on whether the weighted sum of the inputs is greater or less than some threshold value. This threshold value can be understood as the perceptrons bias. It turns out that perceptrons can simulate NAND gates and consequently are universal for computation.

- #### What It Takes To Learn
  For a network to learn, it would need to be capable of making changes to its weight and bias which would correspond to changes in its output. In perceptions these changes aren't small. A change in the weights and bias could cause the output to flip from 0 to 1. The result of this is that while fine tuning the network to be better at some aspect of the problem at hand, its behaviour at other aspects is likely to change. So it seems difficult to gradually modify weights and biases in a perceptron so the network gets closer, generally speaking, to desired behaviour. But that's where Sigmoid Neurons come in.

- #### The Sigmoid Neuron
  The Sigmoid Neuron is much like a perceptron but uses a logistic function for activation. Its input values, rather than being binary, can be anything between 0 and 1. The output of a sigmoid neuron is also non binary. Rather it approximates a perceptron such that the output is 1 when w.x + b is large, 0 when w.x + b is very negative and anything in between when w.x + b is of a modest size. w and x are vectors whose components are the weights and inputs, respectively. And since the resultant shape of the sigmoid function is smooth, we can figure out how change in weight and bias will affect change in the output by computing some partial derivatives.

- #### Final Neural Network
  Our neural network is a bunch of fully connected sigmoid neurons. The input layer contains 784 neurons, each corresponding to a pixel in our 28 by 28 pixel image. The input pixels are freyscale with a value of 0.0 for white and 1.0 for black and in between valus for gradually darkening shdes of grey. Our network will have some hidden layers of size n, which we would experiment to find a good size on and an output layer of size 10, corresponding with digits 0 to 9.

---

### Data Acquisition and Preprocessing

To recognise digits, we first need a data set to train from. I used the MNIST dataset. The MNIST dataset has 60,000 images that can be used for training and 10,000 that can be used for test.. Images were normalized to ensure consistency in data representation, crucial for effective learning.

---

### Training and Optimization

To learn, we need to adjust the weights and bias. To do this, we need a cost function to tell us how well we are achieving our goal. This cost function we use is the mean squared error. The smaller the error, the better our results. So our training will strive to minimize the cost through a technique called gradient descent. 

---

### Conclusion

This project has been invaluable to getting a good grasp on neural networks. But it is far from finished as Michael's book goes all the way to deep neural networks and has lots of excercises I am yet to complete. Will be updating the repository as I work through the book and learn more concepts.  

---

### Credits

Special thanks to:
- **[@MichaelNielsen](https://github.com/mnielsen)**: For his invaluable book "Neural Networks and Deep Learning."
- **[@3Blue1Brown](https://github.com/3b1b)**: For insightful tutorial series on Neural Networks (see [here](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=hxYwfbFWpA5Jp9dr)).
- **[@NeuralNine](https://github.com/NeuralNine)**: For comprehensive tutorials on building neural networks with TensorFlow and Keras (see [here](https://youtu.be/bte8Er0QhDg?si=mwkLqkwNfLtzBBMh)).
- **[@Bot-Academy](https://github.com/Bot-Academy)**: For providing simple yet effective code examples.
