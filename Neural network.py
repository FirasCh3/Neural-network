import numpy as np
import pandas as pd


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def forward_propagation(input):
    return True
df=pd.read_csv("data/mnist_train.csv")
labels=df["label"]
labels=np.array(labels[:10])  #taking 10 samples of labels from the dataset
inputs=df.drop(columns=["label"])  #dropping the label column
inputs=np.array(inputs[:10])  #taking 10 samples of inputs (pixels) from the dataset
iterations = 10  #number of epochs (forward propagation+backward propagation)
np.random.seed(0)  #use the same random numbers on each execution
weights = np.random.randn(3, 784) #generating random weights in a matrix 3 by 784 which means we have 3 nodes in the first layer
biases1 = np.zeros((3, 1), dtype=float)
biases2 = np.zeros((10, 1), dtype=float)
weights2 = np.random.randn(10,3)
def get_label_matrix(x):
    label = np.zeros((10,1),dtype=int)
    label[x,0] = 1 #gennerate a matrix that has zeros in all it's rows except in the x row
    return label


#for i in range(iterations):
#put all of this in a forward_propagation function
for input,label in zip(inputs,labels):
    expected_output=get_label_matrix(label)
    input.shape += (1,) #each input in inputs used to be a vector we modified its shape to make it a matrix with 784 rows and 1 column
    layer1 = sigmoid(weights.dot(input)+biases1) #layer 1 is calculated then we apply sigmoid as an activation function which tells us if the neuron is active or not
    outputs = sigmoid(weights2.dot(layer1)+biases2)  #calculate the layer of output
    mean_squared_error = (1/np.shape(outputs)[0]) * np.sum((expected_output-outputs)**2) #calculating mean squared error which is basically # subtracting the expected output from the
                                                                                         # output squaring it and then summing it and diving it by length of rows

#add backward propagation
print('test')


