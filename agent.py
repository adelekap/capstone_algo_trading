

class InvestorAgent():
    def __int__(self,startingCapital:float,Strategy):
        self.capital_0 = startingCapital
        self.capital_t = startingCapital
        self.strategy = Strategy
        self.gains = 0
        self.positions = []  # will populate with positions once the agent holds one

    def get_indicator(self):
        return self.strategy.arithmetic_returns()