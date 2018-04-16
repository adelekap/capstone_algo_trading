from DueDiligence import sectorSuggestor
from environment import trade
from mongoObjects import CollectionManager

class TradingFramework():
    def __init__(self,start,capital,p=0.015,sharePer=0.5,stop='2018-02-05'):
        self.manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')

        self.dates = self.manager.dates()
        self.startIndex = self.dates.index(start) + 1
        self.stop = stop
        self.stopIndex = self.dates.index(stop)

        self.suggestor = sectorSuggestor(self.startIndex)
        self.suggestor.build_sector_NN()

        self.loss = None
        self.model = None
        self.p = p
        self.sharePer = sharePer
        self.startDate = date
        self.startingCapital = capital


    def run_simulation(self):
        for day in range(self.startIndex,self.stopIndex):
            sectorToInvestIn = self.suggestor.predict(day)
            print(sectorToInvestIn)

    def trade_stock(self,ticker):
        trade(self.loss, self.model, self.p, self.sharePer, self.startDate, self.startingCapital,
              self,self.stop, ticker, plotting=True)


if __name__ == '__main__':
    # User sets start day (sometime in 2017)
    date = '2017-09-05'

    # User says how much money they start with
    startingCapital = 5000

    # Initialize the trading system
    system = TradingFramework(date, startingCapital)
    system.run_simulation()


