
class Position(object):
    def __init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime):
        self.startDate = startDate
        self.ticker = ticker
        self.investment = investment
        self.purchasePrice = purchasePrice
        self.goal = goal
        self.holdTime = holdTime


class Long(Position):
    def __init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,stopLoss):
        Position.__init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime)
        self.stopLoss = stopLoss

    def at_trigger_point(self,price):
        if price >= self.goal:
            return True
        return False

    def trigger(self,price):
        profit = price - self.purchasePrice
        perProfit = profit/self.purchasePrice
        print(profit,perProfit) #Todo:Make action


class Short(Position):
    def at_trigger_point(self,price):
        if price <= self.goal:
            return True
        return False

    def trigger(self):
        pass #Todo:Make action