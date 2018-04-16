from mongoObjects import CollectionManager
from pymongo import MongoClient
from putAndGetData import get_day_stats
from arima import ArimaModel
from putAndGetData import get_closes_highs_lows
import utils
from datetime import datetime as dt
from datetime import timedelta
from putAndGetData import rel_volume
from LSTM import NeuralNet
from SVM import SVM

class Strategy(object):
    def __init__(self, predictionModel, manager, ticker, currentDate, stopLoss, p=0.02, train_size=0.8, patience=10):
        self.predModel = predictionModel
        self.manager = manager
        self.ticker = ticker
        self.currentDate = currentDate
        self.p = p
        self.stopLoss = stopLoss
        self.closes, self.highs, self.lows, self.dates = get_closes_highs_lows(manager, ticker)
        self.train_size = train_size
        self.patience = patience

    def daily_avg_price(self, date, close=None, high=None, low=None):
        if not close:
            close, high, low = get_day_stats(self.manager, self.ticker, date)
        return (close + high + low) / 3.0

    def percent_variation(self, close, *args):
        P = self.predModel(*args)
        return ((P - close) / close)

    def arithmetic_returns(self, k, date):
        Vi = []
        close = self.closes[date]
        predC = []
        predH = []
        predL = []
        for j in range(1, k + 1):
            d = utils.laterDate(self.currentDate, j)
            if type(self.predModel) == NeuralNet:
                predictedAvgPrice = self.predModel.predict(date)
            elif type(self.predModel) == SVM:
                predictedAvgPrice = self.predModel.predict(date)
            else:
                predictedClose = self.predModel.fit(self.closes[:date] + predC)
                predictedHigh = self.predModel.fit(self.highs[:date] + predH)
                predictedLow = self.predModel.fit(self.lows[:date] + predL)
                predictedAvgPrice = self.daily_avg_price(d, predictedClose, predictedHigh, predictedLow)
                predC.append(predictedClose)
                predH.append(predictedHigh)
                predL.append(predictedLow)
            Vi.append((predictedAvgPrice - close) / close)

        significantReturns = [v for v in Vi if abs(v) > self.p]
        return (sum(significantReturns))

    def make_position(self, agent, signal, date, stopLoss, sharePercent=1,posNum=None):
        v = rel_volume(self.manager, self.ticker, date) #higher means higher than average volume
        buyingPower = agent.buying_power(date)
        shareNum = int(v*buyingPower*sharePercent)
        if shareNum == 0:
            return None
        if signal == 1:
            return agent.long(shareNum, date, stopLoss,posNum)
        if signal == -1:
            return agent.short(shareNum, date, stopLoss,posNum)

    def close_opposing_positions(self, positions, agent, price):
        numOfPositions = len(positions)
        if numOfPositions <= 1:
            return None
        newPosition = positions[numOfPositions - 1]
        for pos in positions[:numOfPositions - 1]:
            impatient = dt.strptime(newPosition.startDate, "%Y-%m-%d") - dt.strptime(pos.startDate,
                                                                                        "%Y-%m-%d") < timedelta(days=5)
            if type(pos) != type(newPosition) and not impatient:
                pos.sell(agent, price)


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    # tickers = [t.lower() for t in list(pd.read_csv('stocks.csv')['Symbol'])]
    ticker = 'googl'
    k = 5
    days = (365, 450)
    model = ArimaModel(1, 1, 0, ticker)
    stopLoss = 100  # Todo: determine this?
    dates = manager.dates()
    p = 0.01  # Todo: Grid search to determine this
    for d in range(days[0], days[1]):
        date = dates[d]
        strat = Strategy(model, manager, ticker, date, stopLoss, p)
        print(strat.arithmetic_returns(k, d))
