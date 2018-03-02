from strategy import Strategy
from position import Long
from position import Short


class InvestorAgent(object):
    def __init__(self,startingCapital:float,strat:Strategy,dateIndex:int):
        self.capital_0 = startingCapital
        self.capital_t = startingCapital
        self.strategy = strat
        self.gains = 0
        self.positions = []  # will populate with positions once the agent holds one
        self.dateIndex = dateIndex

    def check_price(self,date):
        return self.strategy.daily_avg_price(date)

    def signal(self,k=5):
        threshold = k * 0.5 * self.strategy.p
        T = self.strategy.arithmetic_returns(k,self.dateIndex)
        if abs(T) >= threshold:
            if T > 0:
                return 1
            return -1
        return 0

    def buying_power(self,date):
        price = self.check_price(date)
        return int(self.capital_t/price)

    def long(self,shareNum,date,stopLoss):
        price = self.strategy.daily_avg_price(date)
        investment = shareNum * price
        goal = price + (price * self.strategy.p)
        position = Long(date,self.strategy.ticker,investment,price,goal,self.strategy.patience,stopLoss,shareNum)
        self.positions.append(position)
        self.capital_t -= investment

    def sell(self,position,price):
        sellReturn = position.shares * price
        profit = sellReturn - position.investment
        self.capital_t += sellReturn
        perProfit = profit / position.investment
        print(profit, perProfit)  # Todo:Make action
        self.positions.remove(position)


    def short(self):
        pass #Todo:implement!