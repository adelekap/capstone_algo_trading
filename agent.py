from Strategy.strategy import Strategy
from Strategy.position import Long, Short


class InvestorAgent(object):
    def __init__(self, startingCapital: float, strat: Strategy, dateIndex: int):
        self.capital_0 = startingCapital
        self.capital_t = startingCapital
        self.strategy = strat
        self.gains = 0
        self.positions = []
        self.dateIndex = dateIndex
        self.profits = []
        self.capitalHistory = []
        self.totalAssetHistory = []

    def check_price(self, date):
        """
        Gets the daily average price for a given date.
        :param date: string of the data YYYY-MM-DD
        :return: daily average price
        """
        return self.strategy.daily_avg_price(date)

    def signal(self, T, k=5):
        """
        Whether to open a long position, short position,
        or no position at all.
        :param T: indicator variable
        :param k: number of days
        :return: signal
        """
        threshold = k * 0.5 * self.strategy.p
        if abs(T) > threshold:
            if T > 0:
                return 1
            return -1
        return 0

    def buying_power(self, date):
        """
        The maximum amount of shares the agent can buy.
        :param date: current date YYYY-MM-DD
        :return:
        """
        price = self.check_price(date)
        trueCapital = self.capital_t
        for pos in self.positions:
            if type(pos) == Short:
                trueCapital -= pos.initialInvestment
        power = int(trueCapital / price)
        if power <= 0:
            return 0
        return power

    def long(self, shareNum, date, stopLoss, posNum):
        """
        Opens a long position
        :param shareNum: number of shares
        :param date: date of position opening
        :param stopLoss: stop loss
        :param posNum: position ID number
        :return: string for logging the position
        """
        price = self.strategy.daily_avg_price(date)
        investment = shareNum * price
        goal = price + (price * self.strategy.p)
        position = Long(date, self.strategy.ticker, investment, price, goal, self.strategy.patience, stopLoss, shareNum,
                        posNum)
        self.positions.append(position)
        self.capital_t -= investment
        return f'long,{price},{shareNum},{investment},,{self.capital_t}'

    def short(self, shareNum, date, stopLoss, posNum):
        """
        Opens a short position
        :param shareNum: number of shares
        :param date: date of position opening
        :param stopLoss: stop loss
        :param posNum: position ID number
        :return: string for loggin the position
        """
        borrowPrice = self.strategy.daily_avg_price(date)
        investment = shareNum * borrowPrice
        goalPrice = (1 - self.strategy.p) * borrowPrice
        position = Short(date, self.strategy.ticker, investment, borrowPrice,
                         goalPrice, self.strategy.patience, stopLoss, shareNum, posNum)
        self.positions.append(position)
        self.capital_t += investment
        return f'short,{borrowPrice},{shareNum},{investment},,{self.capital_t}'

    def update_assets(self, update):
        """
        Updates the current assets of the agent.
        :param update: thing to update
        :return: None
        """
        self.totalAssetHistory.append(update)
