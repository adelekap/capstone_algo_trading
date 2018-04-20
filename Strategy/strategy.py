from DataHandler.putAndGetData import get_day_stats
from DataHandler.putAndGetData import get_closes_highs_lows
import utils
from datetime import datetime as dt
from datetime import timedelta
from DataHandler.putAndGetData import rel_volume
from Models.LSTM import NeuralNet
from Models.SVM import SVM


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
        """
        Average price of a given day
        :param date: date of interest
        :param close: closing price
        :param high: high price
        :param low: low price
        :return: average price
        """
        if not close:
            close, high, low = get_day_stats(self.manager, self.ticker, date)
        return (close + high + low) / 3.0

    def percent_variation(self, close, *args):
        """
        Returns the percent change
        :param close: close price
        :param args: arguments for the predictive model
        :return: the percent variation
        """
        P = self.predModel(*args)
        return ((P - close) / close)

    def arithmetic_returns(self, k, date):
        """
        Calculates the arithmetic returns
        :param k: number of days
        :param date: current date
        :return: T
        """
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

    def make_position(self, agent, signal, date, stopLoss, sharePercent=1, posNum=None):
        """
        Opens a position
        :param agent: investor agent
        :param signal: whether to open a short or long position
        :param date: current date
        :param stopLoss: stop loss
        :param sharePercent: amount of shares to buy as a percent of total
        :param posNum: position ID number
        :return: returns the results of the position
        """
        v = rel_volume(self.manager, self.ticker, date)  # higher means higher than average volume
        buyingPower = agent.buying_power(date)
        shareNum = int(v * buyingPower * sharePercent)
        if shareNum == 0:
            return None
        if signal == 1:
            return agent.long(shareNum, date, stopLoss, posNum)
        if signal == -1:
            return agent.short(shareNum, date, stopLoss, posNum)

    def close_opposing_positions(self, positions, agent, price):
        """
        Closes any positions that are opposite of what just predicted
        :param positions: all positions held
        :param agent: investor agent
        :param price: current price
        :return: None
        """
        numOfPositions = len(positions)
        if numOfPositions <= 1:
            return None
        newPosition = positions[numOfPositions - 1]
        for pos in positions[:numOfPositions - 1]:
            impatient = dt.strptime(newPosition.startDate, "%Y-%m-%d") - dt.strptime(pos.startDate,
                                                                                     "%Y-%m-%d") < timedelta(days=5)
            if type(pos) != type(newPosition) and not impatient:
                pos.sell(agent, price)
