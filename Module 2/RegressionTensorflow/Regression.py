import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import models, layers, regularizers, initializers
from tensorflow.keras.utils import plot_model

# --- Loading the dataset ---
boston_housing = datasets.boston_housing.load_data()
features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
(X, y), _ = boston_housing

# Split the data into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --- Scaling the dataset beacuase it contains values extreme values ---
scaler = RobustScaler()
X_train_prep = scaler.fit_transform(X_train)
X_val_prep = scaler.transform(X_val)
X_test_prep = scaler.transform(X_test)


# --- Creating the model ---
model = models.Sequential()
model.add(layers.Dense(13, activation='relu', 
                           input_shape=X_train.shape[1:],
                           kernel_regularizer=regularizers.l2(0.01),
                           kernel_initializer=initializers.HeUniform(),
                           bias_initializer="ones"))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1))


model.compile(loss='mean_squared_error',
              optimizer='sgd',
              )
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# --- Training the model ---
history = model.fit(X_train_prep, 
                      y_train, 
                      epochs=70,
                      validation_data=(X_val, y_val))

# --- Plotting the Loss of the current model ---
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show() 