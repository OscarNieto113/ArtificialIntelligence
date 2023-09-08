import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import models, layers, regularizers, initializers, datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from scipy import stats

"""
    Plot training and validation metrics over epochs.

    Parameters:
        history: History
            The history object returned by model.fit.
"""
def plot_metrics(history):
    frame = pd.DataFrame(history.history)
    epochs = np.arange(len(frame))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(epochs, frame['loss'], label="Train")
    axes[0].plot(epochs, frame['val_loss'], label="Validation")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epochs")
    axes[0].legend()

    # MAE plot
    axes[1].plot(epochs, frame['mean_absolute_error'], label="Train")
    axes[1].plot(epochs, frame['val_mean_absolute_error'], label="Validation")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_title("Mean Absolute Error vs Epochs")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
"""
    Remove outliers from the dataset using Z-scores.
"""
def remove_outliers(X, y, z_score_threshold=3):
    z_scores = np.abs(stats.zscore(X))
    outlier_indices = np.where(z_scores > z_score_threshold)
    X_no_outliers = np.delete(X, outlier_indices[0], axis=0)
    y_no_outliers = np.delete(y, outlier_indices[0])

    return X_no_outliers, y_no_outliers

"""
    Evaluate a regression model and print relevant metrics.
"""
def evaluate_model(model, X_train, y_train, X_val, y_val):

    train_loss = model.evaluate(X_train, y_train)
    val_loss = model.evaluate(X_val, y_val)

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_mae = mean_absolute_error(y_train, train_predictions)
    val_mae = mean_absolute_error(y_val, val_predictions)

    r2 = r2_score(y_val, val_predictions)

    print(f"Training Loss (MSE): {train_loss}")
    print(f"Validation Loss (MSE): {val_loss}")
    print(f"Training MAE: {train_mae}")
    print(f"Validation MAE: {val_mae}")
    print(f"R-squared (R2): {r2}")
    
# Data gattering
boston_housing = datasets.boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

# Scaling the data
scaler = RobustScaler()

X_train_prep = scaler.fit_transform(X_train)
X_val_prep = scaler.transform(X_val)
X_test_prep = scaler.transform(X_test)

# Model 1
model = Sequential()

model.add(layers.Dense(13, activation='relu',
                           input_shape=X_train.shape[1:],
                           kernel_regularizer=regularizers.l2(0.01),
                           kernel_initializer=initializers.HeUniform(),
                           bias_initializer="ones"))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1))


model.compile(loss='mean_squared_error',
              optimizer='sgd',
               metrics=['mean_absolute_error']
)

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Training Model 1
history = model.fit(
    X_train_prep,
    y_train,
    epochs=100,
    validation_data=(X_val_prep, y_val)
)

# Metrics Model 1
plot_metrics(history)
evaluate_model(model, X_train_prep, y_train, X_val_prep, y_val)

# Model improvements
# Removing Outliers
X_train_no_outliers, y_train_no_outliers = remove_outliers(X_train_prep, y_train)
X_test_no_outliers, y_test_no_outliers = remove_outliers(X_test_prep, y_test)
X_val_no_outliers, y_val_no_outliers = remove_outliers(X_val_prep, y_val)

#Model 2
model2 = Sequential()

model2 = models.Sequential()
model2.add(layers.Dense(13, activation='relu',
                       input_shape=X_train.shape[1:],
                       kernel_regularizer=regularizers.l2(0.01),
                       kernel_initializer=initializers.HeUniform(),
                       bias_initializer="ones"))
model2.add(layers.Dense(10, activation='relu'))
model2.add(layers.Dense(1))

model2.compile(loss='mean_squared_error',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])


#plot_model(model2, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#Train Model 2
history2 = model2.fit(
    X_train_no_outliers,
    y_train_no_outliers,
    epochs=100,
    validation_data=(X_val_no_outliers, y_val_no_outliers)
)

# Metrics Model 1
plot_metrics(history2)
evaluate_model(model, X_train_no_outliers, y_train_no_outliers, X_val_no_outliers, y_val_no_outliers)