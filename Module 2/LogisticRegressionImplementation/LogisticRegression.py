import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

"""
    Calculate a confusion matrix based on predicted and actual values.

    Parameters:
    predictions (list): List of predicted labels (0 or 1).
    actual (list): List of actual labels (0 or 1).

    Returns:
    array: A 2x2 confusion matrix where rows represent actual values (0 and 1),
            and columns represent predicted values (0 and 1).
"""
def confusion_matrix(predictions, actual):
    conf_matrix = np.zeros((2, 2))
    for i in range(len(predictions)):
        conf_matrix[predictions[i]][actual[i]] += 1
    return conf_matrix

"""
    Standardize the input data using mean and standard deviation.

    This function takes a dataset as input and standardizes it by subtracting the mean and dividing by the standard
    deviation for each feature.

    Parameters:
    data (numpy.ndarray): The input dataset where each row represents a sample, and each column represents a feature.

    Returns:
    numpy.ndarray: The standardized data.
"""
def fit_transform(data):
    mean = np.mean(data, axis=0) # Calculate the mean and standard deviation for each feature
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    scaled_data = (data - mean) / std # Standardize the data
    return scaled_data

"""
    Calculate the F1 score based on predicted and actual labels.

    Parameters:
    predictions (list): List of predicted labels (0 or 1).
    y_test (list): List of actual labels (0 or 1).

    Returns:
    float: The F1 score, a measure of a model's accuracy that considers both precision and recall.
"""
def f1_score(predictions, y_test):
    tn, fp, fn, tp = confusion_matrix(predictions, y_test).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

"""
    Calculate and print various evaluation metrics based on predicted and actual labels.

    Parameters:
    prediction (list): List of predicted labels (0 or 1).
    y_test (list): List of actual labels (0 or 1).
"""
def get_rates (prediction, y_test):
    tn, fp, fn, tp = confusion_matrix(prediction, y_test).ravel()
    print("Accuracy:", (np.mean(prediction == y_test)))
    print("Misclassification Rate:", ((fp + fn) / (tn + fp + fn + tp)))
    print("True Positive Rate :", (tp / (tp + fn)))
    print("False Positive Rate:", (fp / (tn + fp)))
    print("True Negative Rate:", (tn / (tn + fp)))
    print("Precision:", (tp / (tp + fp)))
    print("Prevalence:", ((tp + fn) / (tn + fp + fn + tp)))
    print("F1 Score:", f1_score(prediction, y_test))
    
def plot_error(training_costs):
    plt.plot(training_costs)
    plt.xlabel("Epochs")
    plt.ylabel("Cost-Error")
    plt.title("Training Cost Over Epochs")
    plt.show()
    
def plot_function(x_train, x_test, y_train, y_test):
    x_range = np.linspace(-3, 3, 100)

    # Calculate the corresponding y-values using the trained model
    y_range = model.sigmoid(np.dot(np.c_[np.ones(len(x_range)), x_range], model.theta))

    plt.scatter(x_train[y_train == 0][:, 0], x_train[y_train == 0][:, 1], color='red', label='Train Class 0')
    plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], color='blue', label='Train Class 1')
    plt.scatter(x_test[y_test == 0][:, 0], x_test[y_test == 0][:, 1], color='orange', marker='x', label='Test Class 0')
    plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], color='green', marker='x', label='Test Class 1')

    plt.plot(x_range, y_range, color='purple', label='Logistic Function')
    plt.xlabel("Age, Estimated Salary")
    plt.ylabel("Purchased")
    plt.legend()
    plt.title("Logistic Regression Model")
    plt.show()

class LogisticRegression:
    """
        Initialize the Logistic Regression model.

        Parameters:
        learning_rate (float): The learning rate determines the adjustment every epoch.
        epochs (int): The number of training epochs.
    """
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = 0.5
        self.theta = 0

    """
        Compute the sigmoid function.

        Parameters:
        z : The input values.

        Returns:
        array: The sigmoid function applied to the input.
    """
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    """
        Compute the cross-entropy error.

        Parameters:
        y : The true labels.
        y_hat : The predicted labels.

        Returns:
        float: The cross-entropy error.
    """
    def cross_entropy_error(self, y, y_hat):
        cost = (-1 / len(y)) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return cost
    
    """
        Perform gradient descent to train the logistic regression model.

        Parameters:
        features : The feature matrix.
        labels : The true labels.

        Returns:
        list: A list containing the cost at each epoch.
    """
    def gradient_descent(self, features, labels):
        num_samples, num_features  = features.shape
        self.theta = np.zeros(num_features)
        costs = []

        for i in range (self.epochs):
            z = np.dot(features, self.theta)
            y_pred = self.sigmoid(z)
            gradient = np.dot(features.T, (y_pred - labels)) / num_samples
            self.theta -= self.learning_rate * gradient
            cost = self.cross_entropy_error(labels, y_pred)
            costs.append(cost)
        return costs

    """
        Train the logistic regression model on the given training data.

        Parameters:
        x_train : The training feature matrix.
        y_train : The training labels.

        Returns:
        list: A list containing the cost at each epoch during training.
    """
    def fit(self, x_train, y_train):
        costs = self.gradient_descent(x_train, y_train)
        return costs

    """
        Make predictions using the trained model on new data.

        Parameters:
        x_test : The feature matrix of the test data.

        Returns:
        array: Predicted labels (0 or 1) based on the model.
    """
    def predict(self, x_test):
        y_pred = self.sigmoid(np.dot(x_test, self.theta))
        return (y_pred >= self.threshold).astype(int)


dataset = pd.read_csv(r'dataset.csv')
x = dataset.iloc[:,[2,3]].values #Age, Estimated Salary
y = dataset.iloc[:, 4].values #Purchased

x_scaled = fit_transform(x) # Fit and transform the training data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=1)

# ------------------------------------Model------------------------------------
model = LogisticRegression(learning_rate=0.03, epochs=1500) #Initialize the model 
training_costs = model.fit(x_train, y_train) #Train the model

# Make predictions
prediction = model.predict(x_test)
#print("Predicted class: \n", prediction)
#print("Real values: \n", y_test)

get_rates(prediction, y_test) #Rates of the current model

# ------------------------------------Plot------------------------------------
plot_error(training_costs)
plot_function(x_train, x_test, y_train, y_test)