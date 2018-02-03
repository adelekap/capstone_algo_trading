import pandas_datareader as pdr
import pandas as pd
from movAvgCrossover import signals,aapl

initial_capital= 100000.0

positions = pd.DataFrame(index=signals.index).fillna(0.0)
positions = pd.DataFrame()
positions['AAPL'] = 100 * signals['signal']

portfolio = positions.multiply(aapl['Adj. Close'], axis=0)
pos_diff = positions.diff()

#the value of the positions or shares you have bought, multiplied by the ‘Adj Close’ price
portfolio['holdings'] = (positions.multiply(aapl['Adj. Close'],axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj. Close'],axis=0)).sum(axis=1).cumsum()


