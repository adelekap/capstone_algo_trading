from mongoObjects import CollectionManager,MongoClient
from strategy import Strategy
from agent import InvestorAgent
from arima import ArimaModel
import argparse

class Environment(object):
    def __init__(self,manager:CollectionManager,agent:InvestorAgent,startDay:int):
        timeperiod = manager.dates()
        self.manager = manager
        self.agent = agent
        self.day = startDay
        self.currentDate = timeperiod[self.day]

    def increment_day(self):
        self.day += 1


if __name__ == '__main__':
    """Arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Arima'], metavar='M',
                        help='predictive model to use',default='Arima',required=False) #Todo: add other choices
    parser.add_argument('--startDate', help='start date YYYY-MM-DD',default='2017-11-15',required=False,type=str)
    parser.add_argument('--startingCapital', help='amount of money to start with',default=5000.00,type=float,required=False)
    parser.add_argument('--loss', help='percent of money you are willing to lose',default=.30,type=float,required=False)
    parser.add_argument('--p',help='percent change to flag',default=0.015,type=float,required=False)
    parser.add_argument('--ticker',help='stock to consider',default='aapl',type=str,required=False)
    args = parser.parse_args()


    """Initialize Environment"""
    # Data
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])

    # Time
    dates = manager.dates()
    currentDate = args.startDate
    startDay = dates.index(currentDate)

    # Predictive Model
    if args.model == 'Arima':
        model = ArimaModel(1, 1, 0, args.ticker)

    # Investor, Strategy and Trading Environment
    stopLoss = (1-args.loss) * args.startingCapital
    tradingStrategy = Strategy(model, manager, args.ticker, currentDate, stopLoss, args.p)
    investor = InvestorAgent(args.startingCapital, tradingStrategy, startDay)
    environment = Environment(manager, investor,startDay)