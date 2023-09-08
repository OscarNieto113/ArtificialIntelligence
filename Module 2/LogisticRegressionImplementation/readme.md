# Logistic Regression from Scratch

This repository contains Python code to implement Logistic Regression from scratch and apply it to a dataset of potential SUV customers. The goal is to predict whether a user will purchase the SUV based on their age, gender, and estimated salary.

## Dataset

The dataset used in this project consists of the following columns:

- **User ID**: A unique identifier for each user.
- **Gender**: The gender of the user (male or female).
- **Age**: The age of the user in years.
- **Estimated Salary**: An estimate of the user's annual salary.
- **Purchased**: The target variable, indicating whether the user purchased the SUV (0 for not purchased, 1 for purchased).

The dataset is used for binary classification, where the task is to predict whether a user is likely to purchase the SUV based on the given features.
You can find this dataset in here: https://www.kaggle.com/datasets/sandragracenelson/user-data

## Implementation Details

### Logistic Regression

The heart of this project is the implementation of Logistic Regression. Here's how it works:

- **Sigmoid Function**: Logistic Regression uses the sigmoid function to transform the linear combination of input features into a probability score between 0 and 1. This function maps any real-valued number to the range [0, 1], making it suitable for binary classification.

- **Cross-Entropy Error**: The cross-entropy error (also known as log loss) is used as the cost function to measure the error between the predicted probabilities and the actual labels. It quantifies how well the model's predictions align with the ground truth.

- **Gradient Descent**: The model is trained using gradient descent. Gradient descent iteratively adjusts the model's parameters (weights) to minimize the cross-entropy error. It calculates the gradient of the error with respect to the parameters and updates them in the direction that reduces the error.

### Training and Prediction

Here's how the training and prediction process works in the code:

- **Training**: The `gradient_descent` function is used to train the logistic regression model. It iteratively computes the cost, updates the model's parameters, and stores the cost values for each epoch. The number of training epochs and the learning rate are customizable parameters.

- **Prediction**: After training, the model can make predictions on new data using the `predict` method. It applies the trained model's parameters to the input data, applies the sigmoid function to obtain probabilities, and converts these probabilities to binary predictions using a predefined threshold (typically 0.5).

## How to Use

### Installation

To run this project, you need to have the following installed:

- **Python Console**
  - Upgrade pip:
    ```bash
    python -m pip install --upgrade pip
    ```

- Python libraries:
  - NumPy
  - pandas
  - matplotlib
  - scikit-learn

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib scikit-learn
```
### Running the Code
1. Clone this repository to your local machine.
2. Open a terminal or command prompt and navigate to the project directory.
3. Run the following command to execute the Logistic Regression code:
```bash
python3 LogisticRegression.py
```

## Author
Oscar Nieto Espitia

