from mongoObjects import CollectionManager,MongoClient
from strategy import Strategy
from agent import InvestorAgent
from arima import ArimaModel
from putAndGetData import avg_price_timeseries
import argparse
import utils
import warnings
import sys


class Environment(object):
    def __init__(self,manager:CollectionManager,agent:InvestorAgent,startDay:int):
        self.timeperiod = manager.dates()
        self.manager = manager
        self.agent = agent
        self.day = startDay
        self.currentDate = self.timeperiod[self.day]

    def increment_day(self,strategy):
        self.day += 1
        self.currentDate = self.timeperiod[self.day]
        strategy.currentDate = self.timeperiod[self.day]

    def update_total_assets(self,agent:InvestorAgent):
        liquid = agent.capital_t
        investments = []
        for pos in agent.positions:
            pos.update_investment(agent,self.currentDate)
            investments.append(pos.currentInvestment)
        agent.update_assets(sum(investments)+liquid)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print('||||||||||||||||||||')

    """Arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Arima'], metavar='M',
                        help='predictive model to use',default='Arima',required=False) #Todo: add other choices
    parser.add_argument('--startDate', help='start date YYYY-MM-DD',default='2017-01-05',required=False,type=str)
    parser.add_argument('--startingCapital', help='amount of money to start with',default=5000.00,type=float,required=False)
    parser.add_argument('--loss', help='percent of money you are willing to lose',default=.30,type=float,required=False)
    parser.add_argument('--p',help='percent change to flag',default=0.03,type=float,required=False)
    parser.add_argument('--ticker',help='stock to consider',default='aapl',type=str,required=False)
    parser.add_argument('--sharePer',help='percent possible shares to buy',default=1.0,type=float,required=False)
    parser.add_argument('--stop', help='stop date YYYY-MM-DD',default='2018-02-05',required=False,type=str)
    args = parser.parse_args()


    """Initialize Environment"""
    # Data
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])

    # Time
    dates = manager.dates()
    currentDate = args.startDate
    startDay = dates.index(currentDate)
    stopDay = dates.index(args.stop)


    # Predictive Model
    if args.model == 'Arima':
        model = ArimaModel(1, 1, 0, args.ticker)

    # Investor, Strategy and Trading Environment
    stopLoss = (1-args.loss) * args.startingCapital
    tradingStrategy = Strategy(model, manager, args.ticker, currentDate, stopLoss, args.p)
    investor = InvestorAgent(args.startingCapital, tradingStrategy, startDay)
    environment = Environment(manager, investor,startDay)

    totalDays = stopDay - startDay
    t = 1
    threshold = 0.1
    # Simulate Trading Environment
    for d in range(startDay,stopDay):
        # If investor is already holding one or more positions
        if len(investor.positions):
            for position in investor.positions:
                actionDay = utils.laterDate(position.startDate,position.holdTime)
                currentPrice = investor.check_price(environment.currentDate)
                if environment.currentDate == actionDay or position.at_trigger_point(currentPrice):
                    position.sell(investor,currentPrice)

        T = investor.strategy.arithmetic_returns(5,environment.day)
        sig = investor.signal(T)
        if sig != 0:
            investor.strategy.make_position(investor,sig,environment.currentDate,stopLoss,args.sharePer)
        environment.update_total_assets(investor)
        environment.increment_day(investor.strategy)
        threshold = utils.progress((t/totalDays),threshold)
        t += 1
        sys.stdout.flush()

    """PLOTTING"""
    actualPrice = avg_price_timeseries(manager,args.ticker,dates[startDay:stopDay])
    gain = str(round(((investor.capitalHistory[len(investor.capitalHistory)-
                                              1]-args.startingCapital)/args.startingCapital)*100))+'%'
    print()
    print(gain)

    utils.plot_capital(investor.totalAssetHistory,dates[startDay:stopDay],args.ticker,actualPrice,gain)

    mdd = utils.MDD(investor.totalAssetHistory)
    print('MDD: '+str(mdd))
