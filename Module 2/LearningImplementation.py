import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r'dataset.csv')
x = dataset.iloc[:,[2,3]].values #Age, Estimated Salary
y = dataset.iloc[:, 4].values #Purchased

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
print(x_train)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Gradient descent
            dw = (1/num_samples) * np.dot(X.T, (predictions - y))
            db = (1/num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        class_predictions = [1 if p > 0.5 else 0 for p in predictions]
        return class_predictions

# Example usage
X = np.array([[2.5, 3.5], [1.5, 2.5], [3.5, 4.5], [1.0, 1.0]])
y = np.array([1, 0, 1, 0])

model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)

test_data = np.array([[2.0, 3.0], [1.0, 2.0]])
predictions = model.predict(test_data)
print(predictions)
