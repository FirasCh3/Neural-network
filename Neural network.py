import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

class LogisticRegressor:

    def __init__(self, epochs=10, alpha=0.01):
        self.epochs = epochs
        self.alpha = alpha
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

        return self.__sigmoid(Z)*(1-self.__sigmoid(Z))
    #using xavier weight init to avoid weight explosion or vanishing gradient
    def __weight_init(self, input_length, output_length):
        np.random.seed(123)
        weight = np.random.normal(0, np.sqrt(2/(input_length+output_length)), (output_length, input_length))
        return weight
    def train(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.3, random_state=42)
        epsilon = 1e-10
        self.biases = np.zeros(shape=(3, 1))
        self.biases2 = np.zeros(shape=(1, 1))
        self.weights = self.__weight_init(15, 3)
        self.weight2 = self.__weight_init(3, 1)
        total_loss = []
        correct_predictions = 0
        for i in range(self.epochs):
            loss = 0
            for x, y in zip(self.x_train, self.y_train):
                x = np.reshape(x, (15, 1))
                layer1 = self.__relu(np.dot(self.weights, x)+self.biases)
                output = self.__sigmoid(np.dot(self.weight2, layer1)+self.biases2)
                if output >= 0.5:
                    prediction = 1
                else:
                    prediction = 0

                if prediction == y:
                    correct_predictions += 1
                loss += (-y * np.log(output+epsilon))-((1-y)*np.log(1-output+epsilon))
                sigma = (output-y) * self.weight2.T
                dw1 = x.T * sigma * self.__derivative_relu(np.dot(self.weights, x)+self.biases)
                dw2 = (output-y) * layer1.T
                db1 = sigma * self.__derivative_relu(np.dot(self.weights, x)+self.biases)
                db2 = output-y
                self.weights = self.weights - self.alpha * dw1
                self.weight2 = self.weight2 - self.alpha * dw2
                self.biases = self.biases - self.alpha * db1
                self.biases2 = self.biases2 - self.alpha * db2
            total_loss.append(loss[0, 0]/len(self.x_train))
        print("training accuracy: "+str(correct_predictions/(self.epochs*len(self.x_train))))
        plt.plot([i for i in range(self.epochs)], total_loss)
        plt.show()

    def predict(self):
        correct_predictions = 0
        total_predictions = 0
        pred = []
        for x, y in zip(self.x_test, self.y_test):
            x = np.reshape(x, (15, 1))
            layer1 = self.__relu(np.dot(self.weights, x)+self.biases)
            output = self.__sigmoid(np.dot(self.weight2, layer1)+self.biases2)
            total_predictions += 1
            if output >= 0.5:
                output = 1
            else:
                output = 0
            pred.append(output)
            if output == y:
                correct_predictions += 1

        sns.heatmap(confusion_matrix(self.y_test, pred), annot=True)
        plt.show()
        print("model accuracy on test data: " + str(correct_predictions/total_predictions))
    def __normalize(self, df, column):
        df[column] = (df[column]-df.min()[column])/df.std()[column]
    def pre_process(self, path):
        df = pd.read_csv(path)
        df["glucose"].fillna(value=df["glucose"].mean(), inplace=True)
        df["BMI"].fillna(value=df["BMI"].mean(), inplace=True)
        df["heartRate"].fillna(value=df["heartRate"].mean(), inplace=True)
        df["totChol"].fillna(value=df["totChol"].mean(), inplace=True)
        df["education"].fillna(value=df["education"].mean(), inplace=True)
        df["cigsPerDay"].fillna(value=df["cigsPerDay"].mean(), inplace=True)
        df["BPMeds"].fillna(value=df["BPMeds"].mean(), inplace=True)
        for column in ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]:
            self.__normalize(df, column)
        self.y = np.array(df["TenYearCHD"])
        self.X = np.array(df.drop(columns=["TenYearCHD"]))


reg = LogisticRegressor(epochs=100)
reg.pre_process("./data/framingham.csv")
reg.train()
reg.predict()




