import numpy as np
import pandas as pd
from mongoObjects import CollectionManager, MongoDocument, MongoClient
from environment import trade
import warnings
import argparse

"""
2D Grid search 
---------------
p: (0.001,0.05,0.001) 50 possible values
sharePer: (0.01,1.0,0.02) 50 possible values


mongoDB
---------------
stock: ticker
p: value of p
sharePer: value of sharePer
MDD: downshare
return: return percentage
"""


def save_results(dict, manager, ticker):
    newDoc = MongoDocument(dict, ticker, [])
    manager.insert(newDoc)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    """Arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Arima'], metavar='M',
                        help='predictive model to use', default='Arima', required=False)  # Todo: add other choices
    parser.add_argument('--startDate', help='start date YYYY-MM-DD', default='2017-01-05', required=False, type=str)
    parser.add_argument('--startingCapital', help='amount of money to start with', default=5000.00, type=float,
                        required=False)
    parser.add_argument('--loss', help='percent of money you are willing to lose', default=.30, type=float,
                        required=False)
    parser.add_argument('--p', help='percent change to flag', default=0.03, type=float, required=False)
    parser.add_argument('--ticker', help='stock to consider', default='aapl', type=str, required=False)
    parser.add_argument('--sharePer', help='percent possible shares to buy', default=1.0, type=float, required=False)
    parser.add_argument('--stop', help='stop date YYYY-MM-DD', default='2018-02-05', required=False, type=str)
    args = parser.parse_args()

    """GRID SEARCH"""
    ps = np.arange(0.001, 0.051, 0.001)
    sharePers = np.arange(0.01, 1.01, 0.02)
    stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]
    manager = CollectionManager('grid_search', MongoClient()['AlgoTradingDB'])

    # for stock in stocks:
    #     data = {'p': 0.01, 'sharePer': 0.05, 'MDD': 0.1, 'Return': 0.1}
    #     save_results(data,manager,stock)

    trade(args)

