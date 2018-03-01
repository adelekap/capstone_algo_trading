
class Position(object):
    def __init__(self,startDate,ticker,investment,purchasePrice,goal):
        self.startDate = startDate
        self.ticker = ticker
        self.investment = investment
        self.purchasePrice = purchasePrice
        self.goal = goal

class Long(Position):
    def __init__(self,stopLoss):
        self.stopLoss = stopLoss

class Short(Position):
    def __init__(self):
        pass