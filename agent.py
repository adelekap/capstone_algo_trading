from strategy import Strategy
from mongoObjects import CollectionManager,MongoClient
from arima import ArimaModel
from position import Position

class InvestorAgent(object):
    def __init__(self,startingCapital:float,strat:Strategy,dateIndex:int):
        self.capital_0 = startingCapital
        self.capital_t = startingCapital
        self.strategy = strat
        self.gains = 0
        self.positions = []  # will populate with positions once the agent holds one
        self.dateIndex = dateIndex

    def signal(self,k):
        threshold = k * 0.5 * self.strategy.p
        T = self.strategy.arithmetic_returns(k,self.dateIndex)
        if abs(T) >= threshold:
            if T > 0:
                return 1
            return -1
        return 0

    def long(self,investment,date,ticker):
        price = self.strategy.daily_avg_price(date)
        goal = price + (price * self.strategy.p)
        position = Position(date,ticker,investment,price,goal)
        self.positions.append(position)


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    dates = manager.dates()
    ticker = 'googl'
    model = ArimaModel(1, 1, 0, ticker)
    currentDate = '2017-11-15'

    day = dates.index(currentDate)
    startingCapital = 1000
    stopLoss = .70 * startingCapital
    p = .1
    tradingStrategy = Strategy(model, manager, ticker, currentDate, stopLoss, .01)
    investor = InvestorAgent(startingCapital, tradingStrategy, day)
    instructions = investor.signal(5)

    investor.long(100,currentDate,ticker)