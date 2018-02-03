import pandas as pd
import pandas_datareader as pdr
import numpy as np
import datetime
import matplotlib.pyplot as plt

def get(tickers, startdate, enddate):
  def data(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map (data, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG','NVDA','BA','NGL','INTC','XXII']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2018, 1, 5))

daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

# Calculate the daily percentage change for `daily_close_px`
daily_pct_change = daily_close_px.pct_change()

daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))
plt.savefig('multiplePercChange.png')

# pd.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1,figsize=(12,12))
# plt.savefig('scatterMatrix.png')
# plt.show()