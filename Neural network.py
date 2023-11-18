import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
iterations = 10000  #number of epochs (forward propagation+backward propagation)
def sigmoid(Z):
   return 1/(1+np.exp(-Z))
def derivative(Z):
    return sigmoid(Z)*(1-sigmoid(Z))
def softmax(Z):
    return np.exp(Z)/(np.sum(np.exp(Z)))
def weight_init(nrow,ncolumns):
    limit = np.sqrt(2/(nrow+ncolumns))
    W = np.random.normal(0, limit, size=(nrow, ncolumns))
    return W
def normalize_input(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    normalized = (data-mean)/std
    return normalized
def init():
    df = pd.read_csv("data/mnist_train.csv")
    labels = df["label"]
    labels = np.array(labels[:100])  #taking 20000 samples of labels from the dataset
    inputs = df.drop(columns=["label"])  #dropping the label column
    inputs = np.array(inputs[:100])  #taking 20000 samples of inputs (pixels) from the dataset
    inputs = normalize_input(inputs)
    np.random.seed(0)  #use the same random numbers on each execution
    weights = weight_init(3, 784) #generating random weights in a matrix 3 by 784 which means we have 3 nodes in the first layer
    biases1 = np.zeros((3, 1), dtype=float)
    biases2 = np.zeros((10, 1), dtype=float)
    weights2 = weight_init(10, 3)
    return inputs, labels,weights, weights2, biases1, biases2
def get_label_matrix(x):
    label = np.zeros((10, 1), dtype=float)
    label[x, 0] = 1 #gennerate a matrix that has zeros in all it's rows except in the x row
    return label
def training(inputs, labels, weights , weights2 , biases1 , biases2):
    correct_predictions = 0
    total_predictions = 0
    for i in range(iterations):
        for input, label in zip(inputs, labels):
            expected_output = get_label_matrix(label)
            input.shape += (1,) #each input in inputs used to be a vector we modified its shape to make it a matrix with 784 rows and 1 column
            layer1 = sigmoid(weights.dot(input)+biases1) #layer 1 is calculated then we apply sigmoid as an activation function which tells us if the neuron is active or not
            outputs = sigmoid(weights2.dot(layer1)+biases2)  #calculate the layer of output
            total_predictions += 1
            #calculating derivative of the cost function with respect to the weights2 and biases2
            correct_predictions += int(np.argmax(outputs) == np.argmax(expected_output))
            delta = (-(expected_output-outputs))*derivative(weights2.dot(layer1)+biases2)
            dw2 = delta.dot(layer1.T)
            weights2 = weights2 - 0.1*dw2
            biases2 = biases2 - 0.1*delta
            #derivative of cost function with respect to weights and biases1
            dw1 = ((weights2.T.dot(delta))*derivative(weights.dot(input)+biases1)).dot(input.T)
            weights = weights - 0.1*dw1
            db1 = ((weights2.T.dot(delta))*derivative(weights.dot(input)+biases1))
            biases1 = biases1 - 0.1*db1
        print("finished an epoch --------------------------")
    print(total_predictions, correct_predictions)
    print((correct_predictions/total_predictions)*100)
    return weights,weights2,biases1,biases2
def testing(weights,weights2,biases1,biases2):
    df=pd.read_csv("data/mnist_test.csv")
    labels = df["label"]
    labels = np.array(labels)
    inputs = df.drop(columns=["label"])
    inputs = np.array(inputs)
    inputs = normalize_input(inputs)
    total_predictions = 0
    correct_predictions = 0
    for input, label in zip(inputs, labels):
        total_predictions+=1
        expected_output = get_label_matrix(label)
        input.shape += (1,)
        layer1 = sigmoid(weights.dot(input)+biases1)
        outputs = sigmoid(weights2.dot(layer1)+biases2)
        correct_predictions += int(np.argmax(outputs) == np.argmax(expected_output))
        #print('expected: '+str(label)+" output: "+str(np.argmax(outputs)))
    print(total_predictions/correct_predictions)
inputs, labels, weights, weights2, biases1, biases2 = init()
weights, weights2, biases1, biases2 = training(inputs, labels, weights, weights2, biases1, biases2)
testing(weights, weights2, biases1, biases2)
'''implement multi class classification stuff'''
