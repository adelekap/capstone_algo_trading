from strategy import Strategy
from position import Position


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

    def signal(self,k):
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

    def long(self,shareNum,date):
        price = self.strategy.daily_avg_price(date)
        investment = shareNum * price
        goal = price + (price * self.strategy.p)
        position = Position(date,self.strategy.ticker,investment,price,goal,self.strategy.patience)
        self.positions.append(position)
        self.capital_t -= investment