
class Position(object):
    def __init__(self,startDate:str,ticker:str,investment:float,purchasePrice:float,goal:float,holdTime:int,shares:int,posID):
        self.startDate = startDate
        self.ticker = ticker
        self.initialInvestment = investment
        self.purchasePrice = purchasePrice
        self.goal = goal
        self.holdTime = holdTime
        self.shares = shares
        self.id = posID


class Long(Position):
    def __init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,stopLoss,shares,posID):
        Position.__init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,shares,posID)
        self.stopLoss = stopLoss
        self.currentInvestment = investment

    def at_trigger_point(self,price):
        if price >= self.goal:
            return True
        return False

    def sell(self,agent,price):
        sellReturn = self.shares * price
        agent.capital_t += sellReturn
        agent.positions.remove(self)
        prof = (sellReturn-self.initialInvestment)/self.initialInvestment
        agent.profits.append(prof)
        agent.capitalHistory.append(agent.capital_t)
        return f'long,{price},{self.shares},,{prof},{agent.capital_t}'

    def update_investment(self,agent,date):
        price = agent.check_price(date)
        self.currentInvestment = price * self.shares

class Short(Position):
    def __init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,stopLoss,shares,posID):
        Position.__init__(self,startDate,ticker,investment,purchasePrice,goal,holdTime,shares,posID)
        self.stopLoss = stopLoss
        self.currentInvestment = 0

    def at_trigger_point(self,price):
        if price <= self.goal:
            return True
        return False

    def profit(self,price):
        return self.initialInvestment - price*self.shares

    def sell(self,agent,price):
        profit = self.profit(price)
        agent.capital_t += profit
        agent.capital_t -= self.initialInvestment
        agent.profits.append(profit)
        agent.capitalHistory.append(agent.capital_t)
        agent.positions.remove(self)
        return f'short,{price},{self.shares},,{profit},{agent.capital_t}'

    def update_investment(self,agent,date):
        price = agent.check_price(date)
        self.currentInvestment = self.initialInvestment - price*self.shares
