# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:51:12 2022

@author: Ghita Belaid
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import norm
import time

# Start Time
start_time = time.time()

#Importing stock data
ticker = 'ATVI'

#Even though the IPO was sept 9, 2011, the earliest available data for stocks was
#March 2, 2017 so be wary of that and edit the print statements at the end of the program accordingly
data = pd.DataFrame()
data[ticker] = wb.DataReader(ticker, data_source = 'yahoo', start = '1993-10-25', end = '2020-3-10')['Adj Close']


#recent[-1] is today's stock price
recent = pd.DataFrame()
recent = wb.DataReader(ticker, data_source = 'yahoo', start = '12-12-4')['Adj Close']

#This plots the stock prices
data.plot(figsize=(15,6))
plt.show()

#Plot shows the log returns
log_returns = np.log(1 + data.pct_change())
log_returns.plot(figsize=(15,6))
plt.show()

#Plot shows the log return distribution
seaborn.distplot(log_returns.iloc[1:])
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

"""
Brownian Motion:
r = drift + stdev * e^r

"""

"""
STEP 1: Compute drift which is avg log return + half its variance
"""
# below variables are the mean and variance of all previous returns
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var)



"""
STEP 2: Compute Variance and expected daily return
"""
#1st component in calculating variance:
stdev = log_returns.std()

#2nd component is the random variable (Z) which is our expected return based on the distance between the mean and simulated events,
# expressed in the number of standard deviations.

days =  1000 #in days
simulations = 10000
# "Z corresponds to the distance between the mean and the events, expressed as the number of standard deviations.
Z = norm.ppf(np.random.rand(days, simulations)) 


# Final Brownian motion equation
daily_returns = np.exp(drift.values + stdev.values * Z)

# The start price of the stock (in our case 1,000 days ago)
S0 = data.iloc[0]
#The actual price of the stock after x amount of days 

 
#Creates an empty array the size of the array with expected returns (I think)
stock_price_list = np.zeros_like(daily_returns)
stock_price_list[0] = S0

sum_of_simulated_returns = 0 

for i in range(0, simulations):
    for t in range(1, days):
        # Equation for current stock price
        # Stock = Stock_Price_Of_Yesterday * Expected return
        stock_price_list[t] = stock_price_list[t-1]*daily_returns[t]
        if t == days - 1:
            sum_of_simulated_returns += stock_price_list[t][i]


    

plt.plot(stock_price_list)
print("Oldest available stock price date: October 25, 1993 (10,633 days ago)")
print()
print("Data for calculated for mean and variance of returns taken up until March 10, 2020 (1000 days ago)")
print()
print("The closing stock price on March 10, 2020:", data.iloc[-1])
print("The actual stock price today", recent[-1])
print()
print("Simulations: ", simulations)
print("Estimated Stock Price Return", sum_of_simulated_returns/simulations)
print()
print("The actual return in stock price 1000 days later", recent[-1] - data.iloc[-1])
print()
print()
end_time = time.time()
print("Program runtime:", end_time-start_time, "seconds.")




"""
Snapchat simulation runtime:
"""



















