import numpy as np
import pandas as pd
from mongoObjects import CollectionManager, MongoDocument, MongoClient
from environment import async_grid
import seaborn as sns
import matplotlib.pyplot as plt
from utils import ProgressBar
import multiprocessing as mp
import datetime


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
                    data = self.manager.find({'ticker': self.ticker, 'p': p, 'sharePer': round(sharePer + .01,2)})
                if not len(data):
                    print((p,sharePer))
                for index, row in data.iterrows():
                    ls.append(row[type])
            px = [p for i in ls]
            df['p'] = px
            df['sharePer'] = self.sharePers
            df[type] = ls
            dfs.append(df)
        return pd.concat(dfs)

    def plot_heatmap(self, dataframe, title, cmap,vmin,vmax):
        sns.heatmap(dataframe, cmap=cmap,vmin=vmin,vmax=vmax)
        plt.title(title)
        plt.savefig('plots/GridSearch/{0}_grid_{1}.png'.format(self.ticker, title))
        plt.show()

def grid_search_async(stock):
    jobs = []

    loss = 0.3  # Stop Loss
    model = 'Arima'
    startDate = '2017-09-05'
    startingCapital = 10000
    stop = '2018-02-05'
    ps = [round(i, 3) for i in np.arange(0.001, 0.033, 0.004)]  # 8 options
    sharePers = [round(j, 2) for j in np.arange(0.01, 1.0, 0.05)]  # 12 options
    for p in ps:
        for sharePer in sharePers:
            jobs.append([loss,model,p,sharePer,startDate,startingCapital,stop,stock])

    p = mp.Pool(mp.cpu_count())
    p.map(async_grid, jobs)


if __name__ == '__main__':
    # 96 total
    # ps = [round(i,3) for i in np.arange(0.001, 0.033, 0.004)]  # 8 options
    # sharePers = [round(j,2) for j in np.arange(0.46, 0.98, 0.04)]  # 12 options (waiting for 128)
    # stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]
    # manager = CollectionManager('grid_search_test', 'AlgoTradingDB')
    print(datetime.datetime.now())
    grid_search_async('aapl')
    print(datetime.datetime.now())
    # # progress = ProgressBar(len(ps) * len(sharePers))
    # # for stock in stocks:
    # stock = 'aapl'
    # # progress.initialize()
    # for p in ps:
    #     for sharePer in sharePers:
    #         loss = 0.3  # Stop Loss
    #         model = 'Arima'
    #         startDate = '2017-09-05'
    #         startingCapital = 10000
    #         stop = '2018-02-05'
    #         # result = trade(loss, model, p, sharePer, startDate, startingCapital, stop, stock)
    #         result = async_grid([loss, model, p, sharePer, startDate, startingCapital, stop, stock])
    #         save_results(result, manager, stock)
    #         # progress.progress()