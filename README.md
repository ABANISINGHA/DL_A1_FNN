## CS6910 DL_Assignment_1

## Neural Network Model for Fashion-MNIST Classification

This repository contains code for training and evaluating a neural network model for Fashion-MNIST image classification using various optimization techniques and hyperparameters tuning. The model is implemented in Python using libraries such as NumPy, Matplotlib, and Keras.dataset for data set and sklearn.model_selection for train test split.

Dataset: The Fashion-MNIST dataset consists of 60,000 training images and 10,000 testing images across 10 classes. Each image is a 28x28 grayscale representation of various fashion items. For trianing on the dataset flatten each image and normalise the whole train, valid, and test data between 0 to 1. And converted the labels of each to one hot encoded array of length 10, since there is only 10 class labels.

In this project, we aim to develop a deep learning model to classify Fashion-MNIST images into 10 categories. The model architecture includes multiple hidden layers with different activation functions and optimization algorithms. We also explore hyperparameter tuning using Grid and Bayesian optimization techniques.

## Used optimizers:

Stochastic gradient decent, Momentum based gradient decent, Nestrov, RMSprop, Adam, Nadam

## Initialize Neural Network

For initialization of the Neural network model we need input size, output size, number of hidden layers, number of nodes in the hidden layers and weight (for what kind of weight we are initializing the network). Then NN is initialized. In DL_Assignment1.ipynb the function init_network(args,..).

## Forword Propagation

For forword prop after  initializing the network we need activation function for each hidden layer and softmax function as output activation for classification problem. Then put the data set on the model and predict the output label for the training data.

## Backword propagation

First do forword propagation for each data point to predict the output and from predicted output and actual output calculate the cross entropy loss then by using the backword prop algorithm, for minimizing the loss finding rate of change of loss with respect to the parameters(weights and biases). 

Then by using different optimizers we updating the parameters for the whole training data for a prescribed number of epochs. And give the training loss, training accuracy and validation accuracy. Then by this process model will learn the parameters.

## Model train

Then train the model by using the function model_train for different combination of hyperparameters in Wandb sweep configuration.

## Hyperparameters

            Learning rate: 0.001, 0.0001

            Number of hidden layers: 3, 4, 5

            Number of nodes in each hidden layer: 32, 64, 128

            Activation function: sigmoid, relu, tanh

            Optimization algorithm:  SGD, Adam, RMSprop, mgd, Nadam, nestrov

            Batch size: 16, 32, 64

            Epochs: 5, 10

            Weight initialization: xavier, random

Then find the best configuration of the hyperparameter based on the models validation accuracy.

# Sweep configuration

 How the hyperparameter sweeps were done using WANDB: [Link](https://wandb.ai/wandb_fc/articles/reports/Introduction-to-Hyperparameter-Sweeps-A-Model-Battle-Royale-To-Find-The-Best-Model-In-3-Steps--Vmlldzo1NDQ2Nzk5)
      
      To run a sweep through wandb
      sweep_config - dictionary with sweep parameters
      project name - project name in wandb
      sweep_id = wandb.sweep(sweep_config, proj_name)
      function - Make function to run model_train
      wandb.agent(sweep_id, function,count)
      wandb.finish()
      
      

Method: Grid and Bayes


Metric: Validation accuracy (to be maximised)

## Confusion Matrix

After find the best configuration of the hyperparameter found the test accuracy of the model and plot the confusion matrix with true label and predicted label of the test data. Run the section question 7  in DL_Assignment1.ipynb.

## Prediction on Mnist data set

For learning based on the Fashion-mnist data set choosing three different hyperparameter configuration and predict the test accuracy of the model by using these configuarations on the Mnist data set.

Wandb project report: [link](https://wandb.ai/abanisingha1997/Report%20dl%20assign
ment/reports/MA23M001-Abani-Singha-CS6910-Assignm
ent-1--Vmlldzo3MTc4NTA5)














