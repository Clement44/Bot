# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:43:03 2020

@author: clement
"""


# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
from keras import backend as K
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model

# Variable definition
data_folder = 'Data/' # Folder containing the dataset
n_days = 1640
look_back = 20
# Rolling window forecasting
window_length = 300


# Preallocate the array
dataset = np.empty((n_days, 0))

# Create list to save assets names
assets = []
for f in sorted(os.listdir(data_folder)):

    # Save assets names
    assets.append(f.replace('.csv', ''))

    # Load data
    asset = pd.read_csv(data_folder + f, sep=',', usecols=[2, 3], engine='python')
    asset = asset.values[:n_days]
 
    # Ensure all data is float
    asset = asset.astype('float32')
    dataset = np.append(dataset, asset, axis=1)

mse = []
qlike = []

# Visualize returns and volatility of the first asset
i = 0
plt.plot(dataset[:, 0], label='returns')
plt.plot(dataset[:, 1], label='volatility')
plt.legend()
plt.title(assets[0])
plt.xlabel('Time (days)')
plt.show()

# Normalize data
factor = 2

# Calculate second raw moment
M2 = np.mean(dataset ** 2, axis=0) ** (1/2)

# Apply scaling
dataset_norm = (1/factor) * (dataset / M2)

def create_dataset(dataset, look_back=1):
    """
    Function to convert series from dataset to supervised learning problem
    """
    data_x, data_y = [], []

    for i in range(len(dataset) - look_back):

        # Create sequence of length equal to look_back
        x = dataset[i:(i + look_back), :]
        data_x.append(x)

        # Take just the volatility for the target
        data_y.append(dataset[i + look_back, 1::2])

    return np.array(data_x), np.array(data_y)

# Convert series to supervised learning problem
look_back = 20
X, y = create_dataset(dataset_norm, look_back)

# Declare variables
n_features = dataset.shape[1]
n_assets = y.shape[1]

# Split dataset
training_days = 300
X_train, X_test = X[:training_days], X[training_days:]
y_train, y_test = y[:training_days], y[training_days:]

# Prepare the 3D input vector for the LSTM
X_train = np.reshape(X_train, (-1, look_back, n_features))
X_test = np.reshape(X_test, (-1, look_back, n_features))

batch_size = 1

# Create the model
model = Sequential()
model.add(LSTM(58,
               input_shape=(look_back, n_features),
               batch_size=batch_size,
               stateful=True,
               activity_regularizer=regularizers.l1_l2(),
               recurrent_regularizer=regularizers.l1_l2()))
model.add(Dropout(0.2))
model.add(Dense(n_assets, activation='sigmoid'))

# Compile the LSTM model
model.compile(loss='mse', optimizer='rmsprop')

## Training and evaluating the model (On-line learning)

# Create empty arrays
y_pred = np.empty((0, n_assets))
y_true = np.empty((0, n_assets))

for j in range(training_days - look_back + 1, X.shape[0]):

    if j == (training_days - look_back + 1):

        # First training days for training
        X_train = X[:j]
        y_train = y[:j]

        # Next day for forecasting
        X_test = X[j].reshape(1, look_back, n_features)

        # Ensure the correct shape for LSTM
        X_test = np.tile(X_test, (batch_size, 1, 1))
        y_test = np.tile(y[j], (batch_size, 1))

        # Training epochs
        epochs = 300
    
    else:

        # Available data to refine network state
        X_train = X_test
        y_train = y_test

        # Ensure the correct shape for LSTM
        X_test = X[j].reshape(1, look_back, n_features)
        X_test = np.tile(X_test, (batch_size, 1, 1))
        y_test = np.tile(y[j], (batch_size, 1))

        # Epochs for updating
        epochs = 20
        
    # Fit the model
    for i in range(epochs):
        model.fit(X_train,
                  y_train,
                  epochs=1,
                  batch_size=batch_size,
                  verbose=0,
                  shuffle=False)
        model.reset_states()
    
    # Evaluate the model
    # Make predictions
    predicted_output = model.predict(X_test, batch_size=batch_size)

    predicted_output = predicted_output[0].reshape(1, n_assets)
    true_output = y_test[0].reshape(1, n_assets)

    # Save current prediction into an array
    y_pred = np.append(y_pred, predicted_output, axis=0)
    y_true = np.append(y_true, true_output, axis=0)

# Invert scaling
def invert_standardization(data, M2, factor):
  
    # Consider just volatility series
    M2 = M2[1::2]

    data = factor * data * M2

    return y_pred

# Apply inversion
y_pred = invert_standardization(y_pred, M2, factor)
y_true = invert_standardization(y_true, M2, factor)

def evaluate(y_true, y_pred, folder):
    """
    Function to calculate MSE and QLIKE
    """

    mse = []
    qlike = []

    for i in range(0, 28):
        mse_i = (y_true[:, i] - y_pred[:, i]) ** 2
        qlike_i = np.log(y_pred[:, i]) + (y_true[:, i] /  y_pred[:, i])

        # save results (point by point)
        results = np.array([mse_i, qlike_i]).transpose()
        np.savetxt(folder + assets[i] + '.csv', results, delimiter=',', header='MSE, Q-LIKE', fmt='%10.5f', comments='')

        mse.append(np.mean(mse_i, axis=0))
        qlike.append(np.mean(qlike_i, axis=0))

    return mse, qlike

# Apply EVALUATE function to predictions
mse, qlike = evaluate(y_true, y_pred, '2-OL/')

# save results
results = np.array([mse, qlike]).transpose()
np.savetxt('results/2.csv', results, delimiter=',', header='MSE,Q-LIKE', fmt='%10.5f', comments='')

df = pd.DataFrame({'MSE': mse, 'QLIKE': qlike})
print(df.describe())