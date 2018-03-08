from mongoObjects import CollectionManager,MongoClient
from gridSearch import GridSearch
from utils import plot_heatmap
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    ps = [round(i,3) for i in np.arange(0.001, 0.033, 0.004)]  # 8 options
    sharePers = [round(j,2) for j in np.arange(0.01, 0.46, 0.04)]  # 12 options

    manager = CollectionManager('grid_search', MongoClient()['AlgoTradingDB'])
    grid = GridSearch(manager, 'aapl',ps,sharePers)
    returns = grid.get_grid_search_data('return')
    returns = returns.pivot(index='p', columns='sharePer', values='return')
    plot_heatmap(returns,'Return',"Greens")

    mdds = grid.get_grid_search_data('MDD')
    mdds = mdds.pivot(index='p', columns='sharePer', values='MDD')
    plot_heatmap(mdds,'Drawdown',"Purples")




