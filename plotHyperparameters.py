from mongoObjects import CollectionManager,MongoClient
from gridSearch import GridSearch
from utils import plot_3D
import numpy as np

if __name__ == '__main__':
    ps = [round(i,3) for i in np.arange(0.001, 0.033, 0.004)]  # 8 options
    sharePers = [round(j,2) for j in np.arange(0.01, 0.46, 0.04)]  # 12 options

    manager = CollectionManager('grid_search', MongoClient()['AlgoTradingDB'])
    grid = GridSearch(manager, 'aapl')
    x, y, z = grid.get_grid_search_data('return',ps,sharePers)


    plot_3D(x,y,z,'Returns')
    index = z.index(max(z))
    bestp = x[index]
    bestShareP = y[index]
    print(bestp,bestShareP)

    x,y,z = grid.get_grid_search_data('MDD')
    plot_3D(x,y,z,'Drawdown')
    index = z.index(max(z))
    bestp = x[index]
    bestShareP = y[index]
    print(bestp, bestShareP)

