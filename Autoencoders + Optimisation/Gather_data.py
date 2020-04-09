# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:31:55 2020

@author: clement
"""

import pandas as pd
import numpy as np
import time

from pandas_datareader import data as pdr


# Stocks symbols
df = pd.read_csv('nasdaq100_stocks.csv', header=None)
stocks_symbols = np.squeeze(df, axis=1).tolist()

# Index symbol
index_symbol = '^NDX'

# Dates
start_date = '2013-01-01'
end_date = '2018-12-31'

data = pd.DataFrame()    # Empty dataframe
data[index_symbol] = pdr.DataReader(index_symbol, 'yahoo', start_date, end_date)['Adj Close']


i = 0
while i < len(stocks_symbols):
    print('Downloading.... ', i, stocks_symbols[i])

    try:
        # Use pandas_datareader.data.DataReader to extract the desired data from Yahoo!
        data[stocks_symbols[i]] = pdr.DataReader(stocks_symbols[i], 'yahoo', start_date, end_date)['Adj Close']
        i +=1
        
    except:
        print ('Error with connexion. Wait for 1 minute to try again...')
        # Wait for 30 seconds
        time.sleep(30)
        continue
    
# Remove the missing values from dataframe
data = data.dropna()

# Save data
data.iloc[:, 0].to_pickle('nasdaq100_index_6y.pkl')
data.iloc[:, 1:].to_pickle('nasdaq100_6y.pkl')




