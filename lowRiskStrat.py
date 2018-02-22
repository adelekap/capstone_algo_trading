from mongoObjects import CollectionManager
from pymongo import MongoClient
from putAndGetData import get_day_stats
from arima import fit_arima


class Strategy(object):
    def __init__(self,predictionFunction,manager,ticker,currentDate):
        self.predFunc = predictionFunction
        self.manager = manager
        self.ticker = ticker
        self.currentDate = currentDate

    def daily_avg_price(self):
        close, high, low = get_day_stats(self.manager, self.ticker, self.currentDate)
        return (close + high + low) / 3.0

    def percent_variation(self,close, *args):
        P = self.predFunc(*args)
        return ((P - close) / close)

    def arithmetic_returns(self, k, *args):
        dailyAvgPrice = self.daily_avg_price()
        Vi = []
        for j in range(1, k + 1):
            d = laterDate(self.currentDate, j)
            predictedPrice = self.predFunc(*args)
            Vi.append(self.daily_avg_price(self.manager, self.ticker, d))


class LowRisk(Strategy):
    def __init__(self,p,stopLoss,days):
        self.p = p
        self.stopLoss = stopLoss
        self.days = days



def laterDate(date,j):
    return date #ToDo: return later date k days later




if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    ticker = 'googl'
    date = '2017-02-02'
    strat = Strategy(fit_arima,manager,ticker,date)
    strat.arithmetic_returns()
    print('debug')