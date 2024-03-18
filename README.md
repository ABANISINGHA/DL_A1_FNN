## CS6910 DL_Assignment_1

Neural Network Model for Fashion-MNIST Classification

This repository contains code for training and evaluating a neural network model for Fashion-MNIST image classification using various optimization techniques and hyperparameters tuning. The model is implemented in Python using libraries such as NumPy, Matplotlib, and Keras.dataset for data set and sklearn.model_selection for train test split..

Dataset: The Fashion-MNIST dataset consists of 60,000 training images and 10,000 testing images across 10 classes. Each image is a 28x28 grayscale representation of various fashion items. For trianing on the dataset flatten each image and normalise the whole train, valid, and test data between 0 to 1. And converted the labels of each to one hot encoded array of length 10, since there is only 10 class labels.

In this project, we aim to develop a deep learning model to classify Fashion-MNIST images into 10 categories. The model architecture includes multiple hidden layers with different activation functions and optimization algorithms. We also explore hyperparameter tuning using Grid and Bayesian optimization techniques.

Used optimizers: Stochastic gradient decent, Momentum based gradient decent, Nestrov, RMSprop, Adam, Nadam

For initialization of the Neural network model we need input size, output size, number of hidden layers, number of nodes in the hidden layers and weight (for what kind of weight we are initializing the network). Then NN is initialized. In DL_Assignment1.ipynb the function init_network(args,..).

Ford










