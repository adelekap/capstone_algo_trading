
class Position(object):
    def __init__(self,startDate:str,ticker:str,investment:float,purchasePrice:float,goal:float,holdTime:int,shares:int):
        self.startDate = startDate
        self.ticker = ticker
        self.investment = investment
        self.purchasePrice = purchasePrice
        self.goal = goal
        self.holdTime = holdTime
        self.shares = shares


class Long(Position):
    def __init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,stopLoss,shares):
        Position.__init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,shares)
        self.stopLoss = stopLoss

    def at_trigger_point(self,price):
        if price >= self.goal:
            return True
        return False



class Short(Position):
    def at_trigger_point(self,price):
        if price <= self.goal:
            return True
        return False

    def trigger(self):
        pass #Todo:Make action