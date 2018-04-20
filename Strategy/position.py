class Position(object):
    def __init__(self, startDate: str, ticker: str, investment: float, purchasePrice: float, goal: float, holdTime: int,
                 shares: int, posID):
        self.startDate = startDate
        self.ticker = ticker
        self.initialInvestment = investment
        self.purchasePrice = purchasePrice
        self.goal = goal
        self.holdTime = holdTime
        self.shares = shares
        self.id = posID


class Long(Position):
    def __init__(self, startDate, ticker, investment, purchasePrice, goal, holdTime, stopLoss, shares, posID):
        Position.__init__(self, startDate, ticker, investment, purchasePrice, goal, holdTime, shares, posID)
        self.stopLoss = stopLoss
        self.currentInvestment = investment

    def at_trigger_point(self, price):
        """
        Checks if the position is at a point of closing
        :param price: current price
        :return: True/False
        """
        if price >= self.goal:
            return True
        return False

    def sell(self, agent, price):
        """
        Closes the position
        :param agent: investor agent
        :param price: price at close point
        :return: returns the results for logging position
        """
        sellReturn = self.shares * price
        agent.capital_t += sellReturn
        agent.positions.remove(self)
        prof = (sellReturn - self.initialInvestment) / self.initialInvestment
        agent.profits.append(prof)
        agent.capitalHistory.append(agent.capital_t)
        return f'long,{price},{self.shares},,{prof},{agent.capital_t}'

    def update_investment(self, agent, date):
        """
        Updates the worth of the investment
        :param agent: investor agent
        :param date: current date
        :return: None
        """
        price = agent.check_price(date)
        self.currentInvestment = price * self.shares


class Short(Position):
    def __init__(self, startDate, ticker, investment, purchasePrice, goal, holdTime, stopLoss, shares, posID):
        Position.__init__(self, startDate, ticker, investment, purchasePrice, goal, holdTime, shares, posID)
        self.stopLoss = stopLoss
        self.currentInvestment = 0

    def at_trigger_point(self, price):
        """
        Checks if the position is at the trigger point
        :param price: current price
        :return: True/False
        """
        if price <= self.goal:
            return True
        return False

    def profit(self, price):
        """
        Calculates the profit of the investment
        :param price: current price
        :return: profit
        """
        return self.initialInvestment - price * self.shares

    def sell(self, agent, price):
        """
        Closes the position
        :param agent: investor agent
        :param price: current price
        :return: the results for logging the position
        """
        profit = self.profit(price)
        agent.capital_t += profit
        agent.capital_t -= self.initialInvestment
        agent.profits.append(profit)
        agent.capitalHistory.append(agent.capital_t)
        agent.positions.remove(self)
        return f'short,{price},{self.shares},,{profit},{agent.capital_t}'

    def update_investment(self, agent, date):
        """
        Updates the worth of the investment
        :param agent: investor agent
        :param date: current date
        :return: None
        """
        price = agent.check_price(date)
        self.currentInvestment = self.initialInvestment - price * self.shares
