"""
The "Hello World" of trading strategies: moving average crossover.
When the price of an asset moves from one side of a moving average to the other this
represents a change in momemntum and can be used as a point of making the decision
to enteror exit the market.

Create two separate Simple Moving Averages (SMA) of a time series with differing lookback periods:
40 days and 100 days. If the short moving average exceeds the long moving average then you go long,
 if the long moving average exceeds the short moving average then you exit.
"""
import pandas as pd
import quandl
import numpy as np
import matplotlib.pyplot as plt

aapl = quandl.get("WIKI/AAPL", start_date="2000-01-01", end_date="2017-01-01")

shortWindow = 40
longWindow = 100

signals = pd.DataFrame(index=aapl.index)
signals['signal'] = 0

signals['short_mavg'] = aapl['Close'].rolling(window=shortWindow, min_periods=1, center=False).mean()
signals['long_mavg'] = aapl['Close'].rolling(window=longWindow, min_periods=1, center=False).mean()

#Create Signals [1 = buy , 0 = sell]
signals['signal'][shortWindow:] = np.where(signals['short_mavg'][shortWindow:] >
                                           signals['long_mavg'][shortWindow:],1.0,0.0)
signals['positions'] = signals['signal'].diff()

#PLOTTING
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Price in $')
aapl['Close'].plot(ax=ax1, color='r', lw=2.,label='Closing Price')
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m',label='buy signals')
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k',label='sell signals')
plt.legend()
plt.savefig('simpleMarketStrat.png')
plt.show()

