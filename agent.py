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

    def buying_power(self,date):
        price = self.strategy.daily_avg_price(date)
        return int(self.capital_t/price)

    def long(self,shareNum,date):
        price = self.strategy.daily_avg_price(date)
        investment = shareNum * price
        goal = price + (price * self.strategy.p)
        position = Position(date,self.strategy.ticker,investment,price,goal)
        self.positions.append(position)
        self.capital_t -= investment


if __name__ == '__main__':
    # # Params to set
    # manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    # dates = manager.dates()
    # ticker = 'googl'
    # model = ArimaModel(1, 1, 0, ticker)
    # currentDate = '2017-11-15'
    # day = dates.index(currentDate)
    # startingCapital = 5000
    # stopLoss = .70 * startingCapital
    # p = .1
    #
    # instructions = investor.signal(5)
    # sharePossibility = investor.buying_power(currentDate)
    # investor.long(4, currentDate)
    # print('test')