# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:20:16 2021

@author: pvghd
"""
import numpy as np
import pandas as pd
from datetime import date
import pandas_datareader as pdr
import matplotlib.pyplot as plt

# How many years/months/days are we looking back?
yy = 5
mm = 0 
dd = 0

# Number of trading days in the specified amount of time.
# There are 252 'trading days' in a year, for markets don't open on weekends.
total_years = yy + dd/365 + mm/12
trading_days = total_years * 252

# Set the end date to today
end = date.today()

# Set the start date to yy/mm/dd
start = date( end.year-yy , end.month-mm , end.day-(dd+1) )

""" We add an extra day to the start date because we calculate the return as 
as the percentage change between two days so the first row doesn't have a 
previous day to calculate the percentage change and gets a 'Not a Number (NaN)'
value. Therefore, we get one extra day to calculate the exact timeframe that 
the user wants and drop the NaN value that comes from that extra day -hance 
the use of the .dropna() function. """

# Portfolio
portfolio = ['ADA-USD', 'BTC-USD','ETH-USD','XRP-USD']

# Fetch the adjusted closing price of the portfolio components
# for the specified timeframe from Yahoo Finance.
adj_price = pdr.DataReader(portfolio,'yahoo',start,end)['Adj Close']

# Calculate return as the percentage change between the current 
# and a prior element.
daily_return = adj_price.pct_change().dropna()

# Mean value of each component weighted by its contribution to the portfolio.
mean_return = daily_return.mean()
cov_matrix = daily_return.cov()  # PAIRWISE covariance matrix of the returns.

trials = 100

results = np.zeros((4+len(portfolio)-1,trials))

annualised_return = mean_return * trading_days
annualised_stdev = cov_matrix * np.sqrt(trading_days)

for i in range(trials):
    #select random weights for portfolio holdings
    weights = np.random.random(len(portfolio))
    #rebalance weights to sum to 1
    weights /= np.sum(weights)
    
    #calculate portfolio return and volatility
    portfolio_return = np.sum(mean_return * weights) * trading_days
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    
    #store results in results array
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2,i] = results[0,i] / results[1,i]

    #iterate through the weight vector and add data to results array
    for j in range(len(weights)):
        results[j+3,i] = weights[j]    

#convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe',portfolio[0],portfolio[1],portfolio[2],portfolio[3]])

#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
#locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

#create scatter plot coloured by Sharpe Ratio
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='magma_r')
plt.colorbar()

#plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],color='b',s=100)
#plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1],min_vol_port[0],color='g',s=100)

print('Portfolio with the highest Sharpe Ratio:\n', max_sharpe_port)
print()
print('Portfolio with the lowest volatility:\n', min_vol_port)