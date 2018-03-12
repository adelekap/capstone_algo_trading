import numpy as np
import pandas as pd
from mongoObjects import CollectionManager, MongoDocument, MongoClient
from environment import trade
import seaborn as sns
import matplotlib.pyplot as plt
from utils import ProgressBar


def save_results(dict, manager, ticker):
    newDoc = MongoDocument(dict, ticker, [])
    manager.insert(newDoc)


class GridSearch(object):
    def __init__(self, manager, ticker, ps, sharePers):
        self.manager = manager
        self.ticker = ticker
        self.ps = ps
        self.sharePers = sharePers

    def get_grid_search_data(self, type):
        """
        :param type: "MDD" or "return"
        :return: x,y,z
        """
        dfs = []
        for p in self.ps:
            df = pd.DataFrame()
            ls = []
            for sharePer in self.sharePers:
                data = self.manager.find({'ticker': self.ticker, 'p': p, 'sharePer': sharePer})
                if not len(data):
                    data = self.manager.find({'ticker': self.ticker, 'p': p, 'sharePer': sharePer + .01})
                for index, row in data.iterrows():
                    ls.append(row[type])
            px = [p for i in ls]
            df['p'] = px
            df['sharePer'] = self.sharePers
            df[type] = ls
            dfs.append(df)
        return pd.concat(dfs)

    def plot_heatmap(self, dataframe, title, cmap):
        sns.heatmap(dataframe, cmap=cmap)
        plt.title(title)
        plt.savefig('plots/GridSearch/{0}_grid_{1}.png'.format(self.ticker, title))
        plt.show()


if __name__ == '__main__':
    # 96 total
    ps = [round(i,3) for i in np.arange(0.001, 0.033, 0.004)]  # 8 options
    sharePers = [round(j,2) for j in np.arange(0.01, 0.46, 0.04)]  # 12 options
    stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]
    manager = CollectionManager('grid_search', MongoClient()['AlgoTradingDB'])

    progress = ProgressBar(len(ps) * len(sharePers))
    # for stock in stocks:
    stock = 't'
    progress.initialize()
    for p in ps:
        for sharePer in sharePers:
            loss = 0.3  # Stop Loss
            model = 'Arima'
            startDate = '2017-09-05'
            startingCapital = 10000
            stop = '2018-02-05'
            result = trade(loss, model, p, sharePer, startDate, startingCapital, stop, stock)
            save_results(result, manager, stock)
            progress.progress()
