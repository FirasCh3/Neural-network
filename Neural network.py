import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

class LogisticRegressor:
    def __init__(self, epochs=10, alpha=0.01):
        self.X = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.weights = None
        self.weight2 = None
        self.biases = None
        self.biases2 = None
    def __sigmoid(self, Z):

       return 1/(1+np.exp(-Z))
    def __relu(self, Z):
        return np.maximum(0, Z)
    def __derivative_relu(self, Z):
        return 1 * (Z > 0)
    def __derivative(self, Z):

        return self.sigmoid(Z)*(1-self.sigmoid(Z))
    #using xavier weight init to avoid weight explosion or vanishing gradient
    def __weight_init(self, input_length, output_length):
        np.random.seed(0)
        weight = np.random.normal(0, np.sqrt(2/(input_length+output_length)), (output_length, input_length))
        return weight
    def train(self):
        epsilon = 1e-5
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.3, random_state=42)
        self.biases = np.zeros(shape=(3, 1))
        self.biases2 = np.zeros(shape=(1, 1))
        self.weights = self.__weight_init(7, 3)
        self.weight2 = self.__weight_init(3, 1)
        loss = 0
        i = 0
        for x, y in zip(self.x_train, self.y_train):
            i += 1
            x = np.reshape(x, (7, 1))
            layer1 = self.__relu(np.dot(self.weights, x)+self.biases)
            output = self.__sigmoid(np.dot(self.weight2, layer1)+self.biases2)
            loss += (-y * np.log(output+epsilon))+((1-y)*np.log(1-output+epsilon))
            sigma = (-y+output)*self.weight2
            print(self.weights.shape)
            print(sigma.shape)
            dw1 = sigma*np.dot(x, self.__derivative_relu(np.dot(self.weights, x)+self.biases).T)
            print(np.dot(x, self.__derivative_relu(np.dot(self.weights, x)+self.biases).T).shape)
            print(dw1.shape)
            dw2 = -y*output*layer1
            db1 = sigma.T*self.__derivative_relu(np.dot(self.weights, x)+self.biases)*self.biases

















    #def predict(self):
    def __normalize(self, df, column):
        df[column] = (df[column]-df.min()[column])/df.std()[column]
    def pre_process(self,path):
        df = pd.read_csv(path)
        df = pd.get_dummies(df, columns=["island"], dtype=int)
        df = df.replace({"Adelie": 1, "Gentoo": 0})
        self.__normalize(df, "body_mass_g")
        df = df.drop(columns=["year"])
        self.y = np.array(df["species"][:10])

        self.X = np.array(df.drop(columns=["species"])[:10])


reg = LogisticRegressor()
reg.pre_process("./data/penguins_binary_classification.csv")
reg.train()




