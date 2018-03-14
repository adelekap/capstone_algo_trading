from mongoObjects import CollectionManager,MongoClient
from gridSearch import GridSearch
import numpy as np
import pandas as pd


def plot_gridSearch_results(manager,ticker):
    ps = [round(i, 3) for i in np.arange(0.001, 0.033, 0.004)]  # 8 options
    sharePers = [round(j, 2) for j in np.arange(0.01, 0.46, 0.04)]  # 12 options
    grid = GridSearch(manager, ticker, ps, sharePers)
    returns = grid.get_grid_search_data('return')
    returns = returns.pivot(index='p', columns='sharePer', values='return')
    grid.plot_heatmap(returns, 'Return', "Greens")

    mdds = grid.get_grid_search_data('MDD')
    mdds = mdds.pivot(index='p', columns='sharePer', values='MDD')
    grid.plot_heatmap(mdds, 'Drawdown', "Purples")


if __name__ == '__main__':
    manager = CollectionManager('grid_search', MongoClient()['AlgoTradingDB'])
    stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]

    stocks = ['msft']
    for stock in stocks:
        plot_gridSearch_results(manager,stock)