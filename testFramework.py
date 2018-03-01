from agent import InvestorAgent
from strategy import Strategy
from arima import ArimaModel
from mongoObjects import CollectionManager,MongoClient


manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
dates = manager.dates()
ticker = 'googl'
model = ArimaModel(1, 1, 0, ticker)
currentDate = '2017-11-15'

day = dates.index(currentDate)
startingCapital = 1000
stopLoss = .70 * startingCapital
p = .1
tradingStrategy = Strategy(model,manager,ticker,currentDate,stopLoss,.1)
investor = InvestorAgent(startingCapital,tradingStrategy)

