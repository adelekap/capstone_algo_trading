from ticker import *

aapl = pdr.get_data_yahoo('AAPL',
                          start=datetime.datetime(2006, 10, 1),
                          end=datetime.datetime(2017, 1, 1)).dropna()

aapl['42 day Moving Average'] = aapl.rolling(window=40).mean()['Adj Close']
aapl['252 day Moving Average'] = aapl.rolling(window=252).mean()['Adj Close']

# Plot the adjusted closing price, the short and long windows of rolling means
aapl[['Adj Close', '42 day Moving Average', '252 day Moving Average']].plot()
plt.title('AAPL Adjusted Closing Prices')
plt.savefig('MovingAvg.png')

# Calculate the volatility
min_periods = 75  # Define the minumum of periods to consider
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)

# Plot the volatility
vol.plot(figsize=(10, 8))
plt.ylabel('Moving Historical Volatility')
plt.savefig('volatility.png')

# Show the plot
plt.show()