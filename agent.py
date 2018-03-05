from strategy import Strategy
from position import Long
from position import Short


class InvestorAgent(object):
    def __init__(self,startingCapital:float,strat:Strategy,dateIndex:int):
        self.capital_0 = startingCapital
        self.capital_t = startingCapital
        self.strategy = strat
        self.gains = 0
        self.positions = []
        self.dateIndex = dateIndex
        self.profits = []
        self.capitalHistory = []
        self.totalAssetHistory = []

    def check_price(self,date):
        return self.strategy.daily_avg_price(date)

    def signal(self,T,k=5):
        threshold = k * 0.5 * self.strategy.p
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
        profit = sellReturn - position.initialInvestment
        self.capital_t += sellReturn
        percentProfit = profit / position.initialInvestment
        self.positions.remove(position)
        self.profits.append(percentProfit)
        self.capitalHistory.append(self.capital_t)

    def update_assets(self,update):
        self.totalAssetHistory.append(update)

    def short(self,shareNum,date,stopLoss):
       borrowPrice = self.strategy.daily_avg_price(date)
       investment = shareNum * borrowPrice
       goalPrice = (1 - self.strategy.p) * borrowPrice
       position = Short(date,self.strategy.ticker,investment,borrowPrice,
                        goalPrice,self.strategy.patience,stopLoss,shareNum)
       self.positions.append(position)
       self.capital_t -= investment