from mongoObjects import CollectionManager
from pymongo import MongoClient
from putAndGetData import get_day_stats
from arima import ArimaModel
from putAndGetData import get_closes_highs_lows
import utils

class Strategy(object):
    def __init__(self,predictionModel,manager,ticker,currentDate,stopLoss,p=0.02,train_size=0.8,patience=10):
        self.predModel = predictionModel
        self.manager = manager
        self.ticker = ticker
        self.currentDate = currentDate
        self.p = p
        self.stopLoss = stopLoss
        self.closes, self.highs, self.lows, self.dates = get_closes_highs_lows(manager, ticker)
        self.train_size=train_size
        self.patience = patience

    def daily_avg_price(self,date,close=None,high=None,low=None):
        if not close:
            close, high, low = get_day_stats(self.manager, self.ticker, date)
        return (close + high + low) / 3.0

    def percent_variation(self,close, *args):
        P = self.predModel(*args)
        return ((P - close) / close)

    def arithmetic_returns(self, k,date):
        Vi = []
        close = self.closes[date]
        predC = []
        predH = []
        predL = []
        for j in range(1, k + 1):
            d = utils.laterDate(self.currentDate, j)
            predictedClose = self.predModel.fit(self.closes[:date]+predC)
            predictedHigh = self.predModel.fit(self.highs[:date]+predH)
            predictedLow = self.predModel.fit(self.lows[:date]+predL)
            predictedAvgPrice = self.daily_avg_price(d,predictedClose,predictedHigh,predictedLow)
            Vi.append((predictedAvgPrice-close)/close)
            predC.append(predictedClose)
            predH.append(predictedHigh)
            predL.append(predictedLow)
        significantReturns = [v for v in Vi if abs(v) > self.p]
        return(sum(significantReturns))

    def make_position(self,agent,signal,date,stopLoss,sharePercent=1):
        shareNum = int(agent.buying_power(date)*sharePercent)
        if shareNum == 0:
            return None
        if signal == 1:
            agent.long(shareNum,date,stopLoss)
        if signal == -1:
            agent.short(shareNum,date,stopLoss)


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