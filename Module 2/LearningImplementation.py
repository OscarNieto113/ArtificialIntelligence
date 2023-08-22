import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()# Create a StandardScaler object

dataset = pd.read_csv(r'dataset.csv')
x = dataset.iloc[:,[2,3]].values #Age, Estimated Salary
y = dataset.iloc[:, 4].values #Purchased

x_scaled = scaler.fit_transform(x) # Fit and transform the training data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)

def confusion_matrix(predictions, actual):
    conf_matrix = np.zeros((2, 2))
    for i in range(len(predictions)):
        conf_matrix[predictions[i]][actual[i]] += 1
    return conf_matrix

class LogisticRegression:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand
        self.threshold = 0.5

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, y, y_pred):
        m = len(y)
        cost = (-1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost
    
    def gradient_descent(self, x, y, learning_rate, epochs):
        m, n = x.shape
        theta = np.zeros(n)
        costs = []

        for _ in range(epochs):
            z = np.dot(x, theta)
            y_pred = self.sigmoid(z)
            gradient = np.dot(X.T, (y_pred - y)) / m
            theta -= learning_rate * gradient
            cost = self.cost_function(y, y_pred)
            costs.append(cost)

        return theta, costs

    def fit(self, x, y):
        m, n = x.shape
        self.theta = np.random.rand(n)
        self.costs = []

        for _ in range(self.epochs):
            z = np.dot(x, self.theta)
            y_pred = self.sigmoid(z)
            gradient = np.dot(x.T, (y_pred - y)) / m
            self.theta -= self.learning_rate * gradient
            cost = self.cost_function(y, y_pred)
            self.costs.append(cost)

    def predict(self, x):
        y_pred = self.sigmoid(np.dot(x, self.theta))
        return (y_pred >= self.threshold).astype(int)

# Create and train the logistic regression model
model = LogisticRegression(learning_rate=0.001, epochs=1000)
model.fit(x_train, y_train)

# Make predictions
prediction = model.predict(x_test)
print("Predicted class: \n", prediction)
print("Real values: \n", y_test)

tn, fp, fn, tp = confusion_matrix(prediction, y_test).ravel()
print("Accuracy:", (np.mean(prediction == y_test)))
print("Misclassification Rate:", ((fp + fn) / (tn + fp + fn + tp)))
print("True Positive Rate :", (tp / (tp + fn)))
print("False Positive Rate:", (fp / (tn + fp)))
print("True Negative Rate:", (tn / (tn + fp)))
print("Precision:", (tp / (tp + fp)))
print("Prevalence:", ((tp + fn) / (tn + fp + fn + tp)))