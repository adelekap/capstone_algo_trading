from mongoObjects import CollectionManager
from pymongo import MongoClient
from putAndGetData import get_day_stats
from arima import ArimaModel
from putAndGetData import get_closes_highs_lows
import datetime
import utils

class Strategy(object):
    def __laterDate(self,date, j):
        ds = [int(d) for d in date.split('-')]
        date = datetime.datetime(ds[0], ds[1], ds[2])
        return date + datetime.timedelta(days=j)


    def __init__(self,predictionModel,manager,ticker,currentDate,stopLoss,p=0.02,train_size=0.8):

        self.predModel = predictionModel
        self.manager = manager
        self.ticker = ticker
        self.currentDate = currentDate
        self.p = p
        self.stopLoss = stopLoss
        self.closes, self.highs, self.lows, self.dates = get_closes_highs_lows(manager, ticker)
        self.train_size=train_size


    def daily_avg_price(self,date,close=None,high=None,low=None):
        if not close:
            close, high, low = get_day_stats(self.manager, self.ticker, date)
        return (close + high + low) / 3.0

    def percent_variation(self,close, *args):
        P = self.predModel(*args)
        return ((P - close) / close)

    def arithmetic_returns(self, k,date):
        Vi = []
        close, high, low = get_day_stats(self.manager, self.ticker, self.currentDate)
        for j in range(1, k + 1):
            d = self.__laterDate(self.currentDate, j)
            predictedClose = self.predModel.fit(self.closes[:date])
            predictedHigh = self.predModel.fit(self.highs[:date])
            predictedLow = self.predModel.fit(self.lows[:date])
            predictedAvgPrice = self.daily_avg_price(d,predictedClose,predictedHigh,predictedLow)
            Vi.append((predictedAvgPrice-close)/close)
        significantReturns = [v for v in Vi if abs(v) > self.p]
        return(sum(significantReturns))



if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    # tickers = [t.lower() for t in list(pd.read_csv('stocks.csv')['Symbol'])]
    ticker = 'googl'
    k=5
    days=(365,450)
    model = ArimaModel(1, 1, 0, ticker)
    stopLoss = 100  # Todo: determine this?
    dates = manager.dates()
    p = 0.01  #Todo: Grid search to determine this
    for d in range(days[0],days[1]):
        date = dates[d]
        strat = Strategy(model,manager,ticker,date,stopLoss,p)
        print(strat.arithmetic_returns(k, d))



    #
    # # Plot distribution of Ts
    # with open('Ts.txt','r')as t:
    #     lines = t.read().replace('/','')
    #     Ts = lines.split(',')
    #     Ts = [float(x) for x in Ts]
    # sns.distplot(Ts,75)
    # plt.xlim(-1,1.5)
    # plt.xticks(np.arange(-1,1.5,.25))
    # plt.title('Arithmetic Returns')
    # plt.xlabel('p%')
    # plt.savefig('/Users/adelekap/Documents/capstone_algo_trading/plots/arithReturnsDist.pdf')
    # plt.show()