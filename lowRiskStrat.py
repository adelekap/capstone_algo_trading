from mongoObjects import CollectionManager
from pymongo import MongoClient
from putAndGetData import get_day_stats
from arima import ArimaModel
from putAndGetData import get_closes_highs_lows
import datetime


class Strategy(object):
    def __init__(self,predictionModel,manager,ticker,currentDate):
        self.predModel = predictionModel
        self.manager = manager
        self.ticker = ticker
        self.currentDate = currentDate

    def daily_avg_price(self,date,close=None,high=None,low=None):
        if not close:
            close, high, low = get_day_stats(self.manager, self.ticker, date)
        return (close + high + low) / 3.0

    def percent_variation(self,close, *args):
        P = self.predModel(*args)
        return ((P - close) / close)

    def arithmetic_returns(self, k,closes,highs,lows,dates,p=.02):
        Vi = []
        close, high, low = get_day_stats(self.manager, self.ticker, self.currentDate)
        for j in range(1, k + 1):
            d = laterDate(self.currentDate, j)
            predictedClose = self.predModel.fit(closes)
            predictedHigh = self.predModel.fit(highs)
            predictedLow = self.predModel.fit(lows)
            predictedAvgPrice = self.daily_avg_price(d,predictedClose,predictedHigh,predictedLow)
            Vi.append((predictedAvgPrice-close)/close)
        print(Vi)

class LowRisk(Strategy):
    def __init__(self,p,stopLoss,days):
        self.p = p
        self.stopLoss = stopLoss
        self.days = days



def laterDate(date,j):
    ds = [int(d) for d in date.split('-')]
    date = datetime.datetime(ds[0],ds[1],ds[2])
    return date + datetime.timedelta(days=j)




if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    ticker = 'googl'

    closes,highs,lows,dates = get_closes_highs_lows(manager,ticker)


    for date in range(365,380):
        d = str(dates[date])[0:10]
        strat = Strategy(ArimaModel(1, 1, 0, ticker), manager, ticker, d)
        strat.arithmetic_returns(1, closes[:date], highs[:date], lows[:date], dates[:date])