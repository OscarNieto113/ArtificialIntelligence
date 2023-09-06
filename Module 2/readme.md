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

## Code Structure

- `logistic_regression.py`: Python code to implement Logistic Regression from scratch.
- `dataset.csv`: The dataset used for training and testing the model.
- `README.md`: This documentation file.

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
4. The script will train the Logistic Regression model and make predictions. It provides evaluation metrics such as accuracy, misclassification rate, true positive rate, false positive rate, true negative rate, precision, prevalence, and F1 score.

5. Additionally, the script includes functions for plotting the training cost over epochs and visualizing the logistic regression model.

## Author
Oscar Nieto Espitia

