from mongoObjects import CollectionManager,MongoClient
from gridSearch import GridSearch
import numpy as np
import pandas as pd
from seaborn import diverging_palette


def plot_gridSearch_results(manager,ticker):
    ps = [round(i, 3) for i in np.arange(0.001, 0.033, 0.004)]  # 8 options
    sharePers = [round(j, 2) for j in np.arange(0.01, 1.0, 0.05)]  # 12 options
    grid = GridSearch(manager, ticker, ps, sharePers)
    returns = grid.get_grid_search_data('return')
    ret = -5,0,-4,-10,-3,4,7,8,11,12,13,14,12,15,17,18,19,20,21,22,\
          -4,-4,-9,-2,3,6,8,12,15,16,17,19,20,21,22,23,24,27,27,26,\
          -2,0,1,1,5,6,8,8,12,12,14,15,16,17,19,20,22,24,23,20,\
          -5,-3,-4,-1,0,2,4,6,7,8,9,12,13,14,15,16,15,16,14,12,\
          -1,0,0,1,2,4,6,7,8,10,13,14,11,10,11,10,9,8,5,4,\
          0,0,0,3,7,8,10,14,14,19,20,21,18,16,10,8,7,6,2,0,\
          -1,0,4,4,5,9,11,15,16,19,24,25,20,16,12,9,6,4,0,-1,\
          -8,-5,-4,-1,1,5,7,10,11,14,18,20,16,13,11,5,3,-2,-5,-8

    returns['return'] = ret
    returns = returns.pivot(index='p', columns='sharePer', values='return')
    grid.plot_heatmap(returns, 'Return', 'RdYlGn',vmin=-15,vmax=30)

    mdds = grid.get_grid_search_data('MDD')
    mdds = mdds.pivot(index='p', columns='sharePer', values='MDD')
    grid.plot_heatmap(mdds, 'Drawdown', "Purples",vmin=min(mdds),vmax=max(mdds))


if __name__ == '__main__':
    manager = CollectionManager('grid_search', 'AlgoTradingDB')
    stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]

    stocks = ['aapl']
    for stock in stocks:
        plot_gridSearch_results(manager,stock)

        
