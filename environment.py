from mongoObjects import CollectionManager, MongoClient
from strategy import Strategy
from agent import InvestorAgent
from arima import ArimaModel
from putAndGetData import avg_price_timeseries
import argparse
import utils
import warnings


class Environment(object):
    def __init__(self, manager: CollectionManager, agent: InvestorAgent, startDay: int):
        self.timeperiod = manager.dates()
        self.manager = manager
        self.agent = agent
        self.day = startDay
        self.currentDate = self.timeperiod[self.day]

    def increment_day(self, strategy):
        self.day += 1
        self.currentDate = self.timeperiod[self.day]
        strategy.currentDate = self.timeperiod[self.day]

    def update_total_assets(self, agent: InvestorAgent):
        liquid = agent.capital_t
        investments = []
        for pos in agent.positions:
            pos.update_investment(agent, self.currentDate)
            investments.append(pos.currentInvestment)
        agent.update_assets(sum(investments) + liquid)


def trade(loss, statsModel, p, sharePer, startDate, startingCapital, stop, ticker, plotting=False):
    """Initialize Environment"""
    # Data
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])

    # Time
    dates = manager.dates()
    currentDate = startDate
    startDay = dates.index(currentDate)
    stopDay = dates.index(stop)
    bar = utils.ProgressBar(stopDay - startDay)

    # Predictive Model
    if statsModel == 'Arima':
        model = ArimaModel(1, 1, 0, ticker)

    # Investor, Strategy and Trading Environment
    stopLoss = (1 - loss) * startingCapital
    tradingStrategy = Strategy(model, manager, ticker, currentDate, stopLoss, p)
    investor = InvestorAgent(startingCapital, tradingStrategy, startDay)
    environment = Environment(manager, investor, startDay)

    # Simulate Trading Environment
    bar.initialize()
    for d in range(startDay, stopDay):
        if len(investor.positions):
            for position in investor.positions:
                currentPrice = investor.check_price(environment.currentDate)
                actionDay = utils.laterDate(position.startDate,
                                            position.holdTime)  # Todo: hyperparameter?
                if d != stopDay-1:
                    if environment.currentDate == actionDay or position.at_trigger_point(currentPrice):
                        position.sell(investor, currentPrice)
                else:
                    position.sell(investor, currentPrice)

        T = investor.strategy.arithmetic_returns(5, environment.day)
        sig = investor.signal(T)
        if sig != 0:
            investor.strategy.make_position(investor, sig, environment.currentDate, stopLoss,
                                            sharePer)  # Todo: if switching dir of position, close opposite dir
            # investor.strategy.close_opposing_positions(investor.positions,investor,investor.check_price(dates[d]))
        environment.update_total_assets(investor)
        if d!= stopDay-1:
            environment.increment_day(investor.strategy)
        bar.progress()

    """PLOTTING"""
    actualPrice = avg_price_timeseries(manager, ticker, dates[startDay:stopDay])
    if not len(investor.capitalHistory):
        expReturn = 0
    else:
        expReturn = round(((investor.capitalHistory[len(
            investor.capitalHistory) - 1] - startingCapital) / startingCapital) * 100)
    gain = str(expReturn) + '%'

    possible = round(((actualPrice[-1] - actualPrice[0]) / actualPrice[0]) * 100, 1)
    mdd = utils.MDD(investor.totalAssetHistory)
    if plotting:
        utils.plot_capital(investor.totalAssetHistory, dates[startDay:stopDay], ticker, actualPrice, gain, mdd,
                           possible)

    results = {'p': p, 'sharePer': sharePer, 'MDD': mdd, 'return': expReturn}
    return (results)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    """Arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Arima'], metavar='M',
                        help='predictive model to use', default='Arima', required=False)  # Todo: add other choices
    parser.add_argument('--startDate', help='start date YYYY-MM-DD', default='2017-01-05', required=False, type=str)
    parser.add_argument('--startingCapital', help='amount of money to start with', default=5000.00, type=float,
                        required=False)
    parser.add_argument('--loss', help='percent of money you are willing to lose', default=.30, type=float,
                        required=False)
    parser.add_argument('--p', help='percent change to flag', default=0.03, type=float, required=False)
    parser.add_argument('--ticker', help='stock to consider', default='aapl', type=str, required=False)
    parser.add_argument('--sharePer', help='percent possible shares to buy', default=1.0, type=float, required=False)
    parser.add_argument('--stop', help='stop date YYYY-MM-DD', default='2018-02-05', required=False, type=str)
    args = parser.parse_args()

    trade(args.loss, args.model, args.p, args.sharePer, args.startDate, args.startingCapital, args.stop, args.ticker,
          plotting=True)
