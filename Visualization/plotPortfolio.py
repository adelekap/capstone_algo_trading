from DataHandler.mongoObjects import CollectionManager
from DataHandler.putAndGetData import avg_price_timeseries
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd


def plot_capital(capital: list, time: list, actual: list, percentGain='',capDates = [],title='capital',model=''):
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in time]
    f, axarr = plt.subplots(2)
    axarr[0].plot(capDates, capital[:len(capDates)], color='blue', label='Investor')
    axarr[0].set_title('{0}: {1}'.format(model, percentGain))
    axarr[1].plot(dates, actual, color='grey')
    axarr[1].set_title('Price of SPY: 5%')
    axarr[0].get_xaxis().set_visible(False)
    plt.xticks(fontsize=9, rotation=45)
    plt.tight_layout()
    # plt.savefig('plots/{0}/{1}_{2}.png'.format(model,title,stock))
    plt.show()

def diff(dataset,interval=1):
    differenced = pd.DataFrame()
    features = list(dataset.columns.values)
    for feature in features:
        series = dataset[feature]
        diff = list()
        for i in range(interval, len(series)):
            value = series[i] - series[i - interval]
            diff.append(value)
        differenced[feature] = diff
    return differenced['price']


startDay = 1153
stopDay = 1258
stocks = ['googl','nvda','vz','wmt']
model = 'LSTM'
manager = CollectionManager('5Y_technicals','AlgoTradingDB')
dates = manager.dates()

actualPrice = avg_price_timeseries(manager, 'spy', dates[startDay:stopDay])
diff_price = pd.DataFrame()
diff_price['price'] = actualPrice
diff_price = list(diff(diff_price))

portfolioTrades = f'/Users/adelekap/Documents/capstone_algo_trading/comparison/{model}/trades/'

capitalDates = []
all = pd.DataFrame()
for stock in stocks:
    stockData = pd.read_csv(portfolioTrades+stock+'.csv').iloc[:(stopDay-startDay-4),:]
    all[stock] = stockData['CurrentCapital']
    capitalDates = stockData['Date'].unique()

possible = round(((actualPrice[-1] - actualPrice[0]) / actualPrice[0]) * 100, 1)

portfolioData=list(all.mean(axis=1))



"""PLOTTING"""
expReturn = round(((portfolioData[len(portfolioData) - 1] - 15000) / 15000) * 100)
gain = str(expReturn) + '%'

possible = round(((actualPrice[-1] - actualPrice[0]) / actualPrice[0]) * 100, 1)
mdd = 0
plot_capital(portfolioData, dates[startDay:stopDay],actualPrice,gain,model=model,capDates=list(capitalDates))