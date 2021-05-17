# -*- coding: utf-8 -*-
"""
Created on Fri May 14 08:23:02 2021

@author: kunal
"""

from neural_netwrok import *
import h5py
import numpy as np
import matplotlib.pyplot as plt

#import the cat dataset to check our neural network model
def load_data():

    #load the data from h5 file into python readable format
    train_dataset = h5py.File('catvnoncat_dataset/rain_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('catvnoncat_dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    #reshape the data inserted into numpy array
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    #return the dataset
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

#taking the input of the sizes of the hidden layers
inputs = []

while True:
    inp = input("Input a dimension or leave blank if you are done giving the dimentions of hidden layer")
    if inp == "":
        break
    inputs.append(int(inp))

#now input those values into the layers_dim list
layers_dims = [12288]

for i in inputs:
    layers_dims.append(i)

layers_dims.append(1)

learning_rate = float(input("Give the value of the learning rate"))

num_iterations = int(input("Enter the number of iterations"))

""" Starting of the defination of classifier """

#initialize the clasifier
classifier = Neural_Network()

#fit the dataset
classifier.Fit(X=train_x,Y=train_y,layers_dim=layers_dims ,learning_rate = learning_rate , num_iterations = num_iterations)

#print the first and last value of cost
print(classifier.cost[0] , classifier.cost[-1])

#predict the dataset
predict_train = classifier.Predict(test_x)

#print the accuracy
print("Accuracy = %s" % str(np.sum(predict_train == test_y)/test_y.shape[1]))

plt.plot(classifier.cost)
