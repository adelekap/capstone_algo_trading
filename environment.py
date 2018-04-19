from mongoObjects import CollectionManager, MongoClient, MongoDocument
from strategy import Strategy
from agent import InvestorAgent
from arima import ArimaModel
from putAndGetData import avg_price_timeseries
import argparse
import utils
import warnings
from LSTM import NeuralNet
from AsyncPython.logger import log
from SVM import SVM
import numpy as np
from logger import Logger


def save_results(dict, manager, ticker):
    """
    Saves the results of trading to the database
    :param dict: results
    :param manager: collection manager
    :param ticker: stock ticker
    :return: None
    """
    newDoc = MongoDocument(dict, ticker, [])
    manager.insert(newDoc)


def sharpe_ratio(portfolio, etf, rp, rf=1.72):
    """
    Calculates the Sharpe Ratio -- a measurement of risk
    :param portfolio: the portfolio
    :param etf: the values of SPY
    :param rp: expected portfolio return
    :param rf: risk free rate
    :return: sharpe ratio
    """
    sigp = np.std(portfolio)
    covariance = np.cov(portfolio, etf)[0][1]

    varETF = np.var(etf)
    beta = covariance / varETF  # volatility
    rx = sigp / beta
    return (rp - rf) / rx


class Environment(object):
    def __init__(self, manager: CollectionManager, agent: InvestorAgent, startDay: int):
        self.timeperiod = manager.dates()
        self.manager = manager
        self.agent = agent
        self.day = startDay
        self.currentDate = self.timeperiod[self.day]

    def increment_day(self, strategy):
        """
        Steps through the days
        :param strategy:
        :return:
        """
        self.day += 1
        self.currentDate = self.timeperiod[self.day]
        strategy.currentDate = self.timeperiod[self.day]

    def update_total_assets(self, agent: InvestorAgent):
        """
        Updates the total assets of an agent in the trading environment
        :param agent: agent object
        :return: None
        """
        liquid = agent.capital_t
        investments = []
        for pos in agent.positions:
            pos.update_investment(agent, self.currentDate)
            investments.append(pos.currentInvestment)
        agent.update_assets(sum(investments) + liquid)


def async_grid(args):
    """
    Runs grid search to see parameter values
    :param args: args for the trade method
    :return: None
    """
    manager = CollectionManager('grid_search', 'AlgoTradingDB')
    try:
        stock = args[-1]
        result = trade(*args)
        log(f"{args} SUCCESS!\n{result}\n")
        save_results(result, manager, stock)

    except Exception as e:
        log(f"{args} failed {e}")

    manager.close()


def trade(loss, statsModel, p, sharePer, startDate, startingCapital, stop, ticker, epochs=1, neurons=1,
          plotting=False):
    """
    TRADES a given stock for a period of time
    :param loss: amount of loss willing to take
    :param statsModel: string of the predictive model to be used
    :param p: p value for the strategy
    :param sharePer: share percentage for the strategy
    :param startDate: date to start YYYY-MM-DD
    :param startingCapital: capital to start trading with
    :param stop: stop date YYYY-MM-DD
    :param ticker: stock ticker
    :param plotting: whether you want to plot or not
    :return: results (dictionary)
    """
    logger = Logger('comparison/{0}/trades/{1}.csv'.format(statsModel, ticker))
    logger.log('Date,Open/Close,PositionNum,Type,Price,Shares,Investment,Profit,CurrentCapital')
    positionOpenNum = 0
    positionCloseNum = 0
    warnings.filterwarnings("ignore")
    """Initialize Environment"""
    # Data
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')

    # Time
    dates = manager.dates()
    currentDate = startDate
    startDay = dates.index(currentDate)
    stopDay = dates.index(stop)
    # bar = utils.ProgressBar(stopDay - startDay)

    results = {'p': p, 'sharePer': sharePer}

    # Predictive Model
    if statsModel == 'Arima':
        model = ArimaModel(1, 1, 0, ticker)
        k = 5
    if statsModel == 'LSTM':
        model = NeuralNet(ticker, manager, startDay)
        model.create_network()
        model.train_network()
        k = 1
    if statsModel == 'SVM':
        model = SVM(100, 0.01, ticker, manager, startDay)
        model.fit()
        k = 5

    # Investor, Strategy and Trading Environment
    stopLoss = (1 - loss) * startingCapital
    tradingStrategy = Strategy(model, manager, ticker, currentDate, stopLoss, p)
    investor = InvestorAgent(startingCapital, tradingStrategy, startDay)
    environment = Environment(manager, investor, startDay)

    # Simulate Trading Environment
    print(f'Starting Trading {ticker}')
    for d in range(startDay, stopDay):
        if len(investor.positions):
            for position in investor.positions:
                currentPrice = investor.check_price(environment.currentDate)
                actionDay = utils.laterDate(position.startDate,
                                            position.holdTime)
                if environment.currentDate == actionDay or position.at_trigger_point(currentPrice):
                    sellMessage = position.sell(investor, currentPrice)
                    logger.log(f'{environment.currentDate},close,{position.id},' + sellMessage)
                    positionCloseNum += 1

        T = investor.strategy.arithmetic_returns(k, environment.day)
        sig = investor.signal(T)
        if sig != 0:
            message = investor.strategy.make_position(investor, sig, environment.currentDate, stopLoss,
                                                      sharePer, posNum=positionOpenNum)
            if message != None:
                logger.log(f'{environment.currentDate},open,{positionOpenNum},' + message)
            positionOpenNum += 1
        environment.update_total_assets(investor)
        if d != stopDay - 1:
            environment.increment_day(investor.strategy)
            # bar.progress()

    """PLOTTING"""
    print('PLOTTING')
    actualPrice = avg_price_timeseries(manager, ticker, dates[startDay:stopDay])
    if not len(investor.capitalHistory):
        expReturn = 0
    else:
        expReturn = round(((investor.totalAssetHistory[len(
            investor.totalAssetHistory) - 1] - startingCapital) / startingCapital) * 100)
    gain = str(expReturn) + '%'

    possible = round(((actualPrice[-1] - actualPrice[0]) / actualPrice[0]) * 100, 1)
    mdd = utils.MDD(investor.totalAssetHistory)
    if plotting:
        utils.plot_capital(investor.totalAssetHistory, dates[startDay:stopDay], ticker, actualPrice, gain, mdd,
                           possible, model=statsModel)
    results['MDD'] = mdd
    results['return'] = expReturn
    results['possible'] = possible

    etf = avg_price_timeseries(manager, 'spy', dates[startDay:stopDay])
    print(f'RETURN ON INVESTMENT: {expReturn}')
    print(f'Number of Positions Opened: {positionOpenNum}')
    print(f'Number of Positions Closed:{positionCloseNum}')
    print(f'Sharpe Ratio = {sharpe_ratio(investor.totalAssetHistory, etf, possible)}')

    manager.close()
    return (results)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    """Arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Arima', 'LSTM', 'SVM'], metavar='M',
                        help='predictive model to use', default='Arima', required=False) 
    parser.add_argument('--startDate', help='start date YYYY-MM-DD', default='2017-09-05', required=False, type=str)
    parser.add_argument('--startingCapital', help='amount of money to start with', default=5000.00, type=float,
                        required=False)
    parser.add_argument('--loss', help='percent of money you are willing to lose', default=.30, type=float,
                        required=False)
    parser.add_argument('--p', help='percent change to flag', default=0.03, type=float, required=False)
    parser.add_argument('--ticker', help='stock to consider', default='aapl', type=str, required=False)
    parser.add_argument('--sharePer', help='percent possible shares to buy', default=1.0, type=float, required=False)
    parser.add_argument('--stop', help='stop date YYYY-MM-DD', default='2018-02-05', required=False, type=str)
    parser.add_argument('--epochs', help='Number of Epochs for NN training', default=10, required=False, type=int)
    parser.add_argument('--neurons', help='Number of neurons', default=4, required=False, type=int)
    args = parser.parse_args()

    trade(args.loss, args.model, args.p, args.sharePer, args.startDate, args.startingCapital, args.stop, args.ticker,
          args.epochs, args.neurons, plotting=True)
