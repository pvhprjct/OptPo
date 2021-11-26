# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:20:16 2021

@author: pvhprjct
"""
import numpy as np
import pandas as pd
from datetime import date
import pandas_datareader as pdr
import matplotlib.pyplot as plt

""" Setup """

investment = 1800 # This can be in any currency
portfolio = ['BAT-USD','LTC-USD','MATIC-USD','ADA-USD','XRP-USD','ETH-USD']
data_source = 'yahoo' # Data source: Yahoo is recomended
# Alternative data sources in https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
# Use the other sources if there is no data on the desired stock.

# How many years/months/days are we looking back?
yy = 8
mm = 0 
dd = 0

trials = 1000


""" Code """

# Number of trading days in the specified amount of time.
# There are 252 'trading days' in a year, for markets don't open on weekends.
total_years = yy + dd/365 + mm/12
trading_days = total_years * 252

# Set the end date to today
end = date.today()
# Set the start date to yy/mm/dd
start = date( end.year-yy , end.month-mm , end.day-(dd+1) )


# Fetch the adjusted closing price of the portfolio components
# for the specified timeframe from Yahoo Finance.
adj_price = pdr.DataReader(portfolio,data_source,start,end)['Adj Close']

# Calculate the daily return as the percentage change between the current 
# and a prior element.
daily_return = adj_price.pct_change() # Units = %

mean_return = daily_return.mean() # Mean value of the returns of each stock.
cov_matrix = daily_return.cov()  # PAIRWISE covariance matrix of the returns.

# Empty array to store the results.
output = np.zeros((3+len(portfolio),trials))

annualised_return = mean_return * trading_days
annualised_stdev = cov_matrix * np.sqrt(trading_days)

for i in range(trials):
    # Random weights for portfolio holdings
    weights = np.random.random(len(portfolio))
    # Normalize weights so that they sum to 1
    weights /= np.sum(weights)
    
    #calculate portfolio return and volatility
    portfolio_return = np.sum(mean_return * weights) * trading_days
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    
    #store results in results array
    output[0,i] = portfolio_return
    output[1,i] = portfolio_std_dev
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    output[2,i] = output[0,i] / output[1,i]

    #iterate through the weight vector and add data to results array
    for j in range(len(weights)):
        output[j+3,i] = weights[j]

# Convert output array to Pandas DataFrame format
dataframe_columns = ['ret', 'stdev', 'sharpe']
dataframe_columns.extend(portfolio)
output_dataframe = pd.DataFrame(output.T,columns=dataframe_columns)
# The .T transposes the output array since the categories are originally rows
# instead of columns.

#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = output_dataframe.iloc[output_dataframe['sharpe'].idxmax()]
#locate positon of portfolio with minimum standard deviation
min_vol_port = output_dataframe.iloc[output_dataframe['stdev'].idxmin()]

#create scatter plot coloured by Sharpe Ratio
plt.scatter(output_dataframe.stdev,output_dataframe.ret,c=output_dataframe.sharpe,cmap='magma_r')
plt.xlabel('Volatility')
plt.ylabel('Returns (%)')
plt.colorbar()

#plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],color='b',s=100)
#plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1],min_vol_port[0],color='g',s=100)


print('Original investment =', investment, '\n')
print('Portfolio with the highest Sharpe Ratio:\n', max_sharpe_port, '\n')
print('That is ($):\n', max_sharpe_port[portfolio].T*investment)
print('\n ------------------------- \n')
print('Portfolio with the lowest volatility:\n', min_vol_port, '\n')
print('That is ($):\n', min_vol_port[portfolio].T*investment)
