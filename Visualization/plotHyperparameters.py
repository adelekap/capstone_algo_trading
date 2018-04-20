from Strategy.gridSearch import GridSearch
import numpy as np


def plot_gridSearch_results(manager, ticker):
    ps = [round(i, 3) for i in np.arange(0.001, 0.033, 0.004)]  # 8 options
    sharePers = [round(j, 2) for j in np.arange(0.01, 1.0, 0.05)]  # 12 options
    grid = GridSearch(manager, ticker, ps, sharePers)
    returns = grid.get_grid_search_data('return')
    returns = returns.pivot(index='p', columns='sharePer', values='return')
    grid.plot_heatmap(returns, 'Return', 'RdYlGn', vmin=-15, vmax=30)

    mdds = grid.get_grid_search_data('MDD')
    mdds = mdds.pivot(index='p', columns='sharePer', values='MDD')
    grid.plot_heatmap(mdds, 'Drawdown', "Purples", vmin=min(mdds), vmax=max(mdds))
