from DueDiligence import SectorSuggestor, StockSuggestor
from environment import trade
from mongoObjects import CollectionManager
import pandas as pd
import warnings

id_to_sector = {0: 'Industrials', 1: 'Health Care', 2: 'Information Technology', 3: 'Consumer Discretionary',
                4: 'Utilities', 5: 'Financials', 6: 'Materials', 7: 'Consumer Stapes', 8: 'Real Estate', 9: 'Energy',
                10: 'Telecommunications Services'}


class TradingFramework():
    def __init__(self, start, capital, model, loss, p=0.015, sharePer=0.5, stop='2018-02-05'):
        self.manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')

        self.dates = self.manager.dates()
        self.startIndex = self.dates.index(start) + 1
        self.stop = stop
        self.stopIndex = self.dates.index(stop)

        self.suggestor = SectorSuggestor(self.startIndex)
        self.suggestor.build_sector_NN()

        self.sectorModels = {}
        self.portfolio = {}

        self.loss = loss
        self.model = model
        self.p = p
        self.sharePer = sharePer
        self.startDate = date
        self.startingCapital = capital

    def run_simulation(self):
        """
        Runs the full trading framework.
        :return: None
        """
        warnings.filterwarnings("ignore")
        sectors =[]
        for day in range(self.startIndex, self.stopIndex):
            print(f'--------DAY: {day - self.startIndex +1}--------')
            # Suggest a Sector
            sectorIDtoInvestIn = self.suggestor.predict_sector(day)
            sectorToInvestIn = id_to_sector[sectorIDtoInvestIn]
            print(f'I suggest investing in the {sectorToInvestIn} sector')
            if len(sectors) != 0:
                if sectors[-1] == sectorToInvestIn:
                    print('Already holding positions in this sector...')
                    sectors.append(sectorToInvestIn)
                    continue
            sectors.append(sectorToInvestIn)
            # Suggest top three stocks in this market
            if sectorToInvestIn not in self.sectorModels:
                stockModel = StockSuggestor(sectorToInvestIn, day, self.dates[day])
                self.sectorModels[sectorToInvestIn] = stockModel
                stockModel.build_network()
            else:
                stockModel = self.sectorModels[sectorToInvestIn]
            stockToInvestIn, untilThisDate = stockModel.predict_stock(self.dates[day])
            print(f'I suggest investing in the following stock: {stockToInvestIn}')
            self.trade_stock(stockToInvestIn.lower(),self.startDate,self.dates[day+50])
        print(f'SECTORS INVESTED IN: {sectors}')
        resultManager = CollectionManager('trading_results','AlgoTradingDB')
        resultManager.insert(self.portfolio)


    def trade_stock(self, ticker,start,stop):
        self.portfolio[ticker] = (trade(self.loss, self.model, self.p, self.sharePer, start, self.startingCapital,
              stop, ticker, plotting=True))

if __name__ == '__main__':
    # User sets start day (sometime in 2017)
    date = '2017-09-05'

    # User says how much money they start with
    startingCapital = 5000

    # User chooses which predictive model to use
    mod = 'SVM'

    # User chooses what percent of money they are willing to lose
    stopLoss = 0.3

    # Initialize the trading system
    system = TradingFramework(date, startingCapital, mod, 0.3)
    system.run_simulation()
