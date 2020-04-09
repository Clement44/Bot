# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:50:47 2020

@author: clement
"""

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers

# Load data
stocks_data = pd.read_pickle('nasdaq100_6y.pkl')
index_data = pd.read_pickle('nasdaq100_index_6y.pkl')
assets_names = stocks_data.columns.values

data_assets = stocks_data
data_index = index_data

print("Stocks data (time series) shape: {shape}".format(shape=stocks_data.shape))
print("Index data (time series) shape: {shape}".format(shape=index_data.shape))

stocks_data.head()


# Split data
n_train = int(data_assets.shape[0]*0.8)

# Stocks data
X_train = data_assets.values[:n_train, :]
X_test = data_assets.values[n_train:, :]

# Index data
index_train = data_index[:n_train]
index_test = data_index[n_train:]

# Normalize data
scaler = MinMaxScaler([0, 1])
# Stocks data
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# Index data
scaler_index = MinMaxScaler([0, 1])
index_train = scaler_index.fit_transform(index_train[:, np.newaxis])
index_test = scaler_index.fit_transform(index_test[:, np.newaxis])


# Generate corrupted series by adding noise with normal distribution
noise_factor = 0.05
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

# Clip corrupter data
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)


# Visualize corrupted data
f, axarr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15,7))

# Asset 1
axarr[0,0].plot(X_train[:, 0])
axarr[0,0].set_title(assets_names[0])
axarr[1,0].plot(X_train_noisy[:, 0])

# Asset 2
axarr[0,1].plot(X_train[:, 1])
axarr[0,1].set_title(assets_names[1])
axarr[1,1].plot(X_train_noisy[:, 1])

# Asset 3
axarr[0,2].plot(X_train[:, 2])
axarr[0,2].set_title(assets_names[2])
axarr[1,2].plot(X_train_noisy[:, 2])

plt.show()
plt.savefig('denoisingAE_noisydata.png', bbox_inches='tight')


## Autoencoder - Keras

# Network hyperparameters
n_inputs = X_train.shape[1]

# Training hyperparameters
epochs = 50
batch_size = 1

# Define model
input = Input(shape=(n_inputs,))
# Encoder Layers
encoded = Dense(4, input_shape=(n_inputs,), activation='relu')(input)
decoded = Dense(n_inputs, activation='sigmoid')(encoded)

# Encoder
encoder = Model(input, encoded)

# Autoencoder
model = Model(input, decoded)


# Compile autoencoder
model.compile(loss='mse', optimizer='adam')
model.summary()


# Fit the model
history = model.fit(X_train_noisy,
                    X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=1
                    )


# Visualize loss history
plt.figure()
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.savefig('images/denoisingAE_loss.png', bbox_inches='tight')



# Evaluate model
score_train = model.evaluate(X_train_noisy, X_train, batch_size=batch_size)
score_test = model.evaluate(X_test_noisy, X_test, batch_size=batch_size)

print('Training MSE: %.8f' %score_train)
print('Training MSE: %.8f' %score_test)


# Obtain reconstruction of the stocks
X_train_pred = model.predict(X_train_noisy)
X_test_pred = model.predict(X_test_noisy)

error = np.mean(np.abs(X_train - X_train_pred)**2, axis=0)
print('Training MSE: %.8f' %np.mean(error))

error_test = np.mean(np.abs(X_test - X_test_pred)**2, axis=0)
print('Testing MSE: %.8f' %np.mean(error_test))


# Sort stocks by reconstruction error
ind = np.argsort(error)
sort_error = error[ind]
sort_assets_names = assets_names[ind]


i = 0
plt.figure()
plt.plot(X_train[:, ind[i]], label=assets_names[ind[i]] + ' Stock')
plt.plot(X_train_pred[:, ind[i]], label=assets_names[ind[i]] + ' AE')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Normalized price')
plt.show()
plt.savefig('images/denoisingAE_' + assets_names[ind[i]] + '.eps', bbox_inches='tight')


# Barplot
plt.figure()
plt.barh(2*np.arange(len(error[:20])), error[ind[:20]], tick_label=assets_names[ind[:20]])
plt.xlabel('MSE')
#plt.xticks(rotation=25)
plt.show()
plt.savefig('images/denoisingAE_MSEbar.png', bbox_inches='tight')


# Identify stocks
n = 5

portfolio_train = X_train_pred[:, ind[:n]]
portfolio_test = X_test_pred[:, ind[:n]]

# Create portfolio in-sample
tracked_index_insample = np.mean(portfolio_train, axis=1)

# Create portfolio out-sample
tracked_index_outofsample = np.mean(portfolio_test, axis=1)

# In-sample
plt.figure()
plt.plot(index_train, label='Nasdaq100 Index')
plt.plot(tracked_index_insample, label='Tracked Index')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Normalized price')
plt.show()
plt.savefig('images/denoisingAE_insample.png', bbox_inches='tight')

# Correlation coefficient (in-sample)
corr_train = np.corrcoef(index_train.squeeze(), tracked_index_insample)[0, 1]
print('Correlation coefficient (in-sample): %.8f' %corr_train)

# Plot tracked index (out-of-sample)
plt.figure()
plt.plot(index_test, label='Nasdaq100 Index')
plt.plot(tracked_index_outofsample, label='Tracked Index')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Normalized price')
plt.show()
plt.savefig('images/denoisingAE_outofsample.png', bbox_inches='tight')

# Correlation coefficient (out-of-sample)
corr_test = np.corrcoef(index_test.squeeze(), tracked_index_outofsample)[0, 1]
print('Correlation coefficient: %.8f' %corr_test)

