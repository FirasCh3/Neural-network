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
        self.biases = None

        self.hidden = np.ndarray(shape=())
    def __sigmoid(self, Z):

       return 1/(1+np.exp(-Z))
    def __derivative(self, Z):

        return self.sigmoid(Z)*(1-self.sigmoid(Z))
    def train(self):
        self.x_train, self.y_train, self.x_test, self.y_test = train_test_split(self.X, self.y, train_size=0.3, random_state=42)
        self.biases = np.zeros(shape=(1, 3))
        self.weights = np.ones(shape=(8, 3))
        for x in self.x_train:
            x = np.reshape(x, (8, 1))
            print(x)
            print(self.weights)
            print(self.biases)
            layer1 = np.dot(x.T, self.weights)+self.biases
            print(layer1)
            print("---------------------------------")






    #def predict(self):
    def pre_process(self,path):
        df = pd.read_csv(path)
        df = pd.get_dummies(df, columns=["island"], dtype=int)
        df = df.replace({"Adelie": 1, "Gentoo": 0})
        sns.pairplot(df)
        self.y = np.array(df["species"][:10])
        self.X = np.array(df.drop(columns=["species"])[:10])


reg = LogisticRegressor()
reg.pre_process("./data/penguins_binary_classification.csv")
reg.train()




