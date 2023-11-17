import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
def derivative(Z):
    return sigmoid(Z)*(1-sigmoid(Z))
df = pd.read_csv("data/mnist_train.csv")
labels = df["label"]
labels = np.array(labels[:100])  #taking 10 samples of labels from the dataset
inputs = df.drop(columns=["label"])  #dropping the label column
inputs = np.array(inputs[:100])  #taking 10 samples of inputs (pixels) from the dataset
iterations = 1000  #number of epochs (forward propagation+backward propagation)
np.random.seed(0)  #use the same random numbers on each execution
weights = np.random.rand(3, 784) #generating random weights in a matrix 3 by 784 which means we have 3 nodes in the first layer
biases1 = np.zeros((3, 1), dtype=float)
biases2 = np.zeros((10, 1), dtype=float)
weights2 = np.random.rand(10, 3)
def get_label_matrix(x):
    label = np.zeros((10, 1), dtype=float)
    label[x, 0] = 1 #gennerate a matrix that has zeros in all it's rows except in the x row
    return label

def training(weights , weights2 , biases1 , biases2):
        #put all of this in a forward_propagation function
    correct_predictions = 0
    total_predictions=0
    for i in range(iterations):
        for input, label in zip(inputs, labels):
            expected_output=get_label_matrix(label)
            input.shape += (1,) #each input in inputs used to be a vector we modified its shape to make it a matrix with 784 rows and 1 column
            layer1 = sigmoid(weights.dot(input)+biases1) #layer 1 is calculated then we apply sigmoid as an activation function which tells us if the neuron is active or not
            outputs = sigmoid(weights2.dot(layer1)+biases2)  #calculate the layer of output
            total_predictions+=1
            mean_squared_error = (1/len(outputs)) * np.sum(np.square(np.subtract(expected_output, outputs))) #calculating mean squared error which is basically # subtracting the expected output from the
                                                                                                             # output squaring it and then summing it and diving it by length of rows
            print(label)
            print(outputs)
            #calculating derivative of the cost function with respect to the weights2 and biases2
            correct_predictions += int(np.argmax(outputs) == np.argmax(expected_output))
            delta = (-(expected_output-outputs))*derivative(weights2.dot(layer1)+biases2)
            delta = 2/len(outputs) * delta
            dw2 = delta.dot(layer1.T)
            weights2 = weights2 - dw2
            biases2 = biases2 - delta
            #derivative of cost function with respect to weights and biases1
            dw1 = ((weights2.T.dot(delta))*derivative(weights.dot(input)+biases1)).dot(input.T)
            weights = weights - dw1
            db1 = ((weights2.T.dot(delta))*derivative(weights.dot(input)+biases1))
            biases1 = biases1 - db1
    print(total_predictions,correct_predictions)
    print((correct_predictions/total_predictions)*100)
training(weights,weights2,biases1,biases2)