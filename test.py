import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aapl = quandl.get("WIKI/AAPL", start_date="2006-10-01", end_date="2012-01-01")

daily_log_returns = np.log(aapl['Adj. Close'].pct_change()+1)

# Plot the distribution of `daily_pct_c`
daily_pct_change = aapl['Adj. Close'] / aapl['Adj. Close'].shift(1) - 1
daily_pct_change.hist(bins=50)
plt.title('Adjusted Closing Percent Change : AAPL')
plt.savefig('perChangeHist.png')


cum_daily_return = (1+daily_pct_change).cumprod().dropna()
plt.plot(list(cum_daily_return))
plt.show()




