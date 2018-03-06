

class Position(object):
    def __init__(self,startDate:str,ticker:str,investment:float,purchasePrice:float,goal:float,holdTime:int,shares:int):
        self.startDate = startDate
        self.ticker = ticker
        self.initialInvestment = investment
        self.currentInvestment = investment
        self.purchasePrice = purchasePrice
        self.goal = goal
        self.holdTime = holdTime
        self.shares = shares

    def update_investment(self,agent,date):
        price = agent.check_price(date)
        self.currentInvestment = price * self.shares


class Long(Position):
    def __init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,stopLoss,shares):
        Position.__init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,shares)
        self.stopLoss = stopLoss

    def at_trigger_point(self,price):
        if price >= self.goal:
            return True
        return False
    def profit(self,price):
        sellReturn = self.shares*price
        profit = sellReturn - self.initialInvestment
        return profit, sellReturn


class Short(Position):
    def __init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,stopLoss,shares):
        Position.__init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,shares)
        self.stopLoss = stopLoss

    def at_trigger_point(self,price):
        if price <= self.goal:
            return True
        return False

    def profit(self,price):
        profit = self.initialInvestment - price*self.shares
        sellReturn = profit
        return profit, sellReturn