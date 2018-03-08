import numpy as np
import pandas as pd
from mongoObjects import CollectionManager, MongoDocument, MongoClient
from environment import trade


def save_results(dict, manager, ticker):
    newDoc = MongoDocument(dict, ticker, [])
    manager.insert(newDoc)


if __name__ == '__main__':
    # 96 total
    ps = np.arange(0.001, 0.033, 0.004)   # 8 options
    sharePers = np.arange(0.01, 0.46, 0.04)  # 12 options
    stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]
    manager = CollectionManager('grid_search', MongoClient()['AlgoTradingDB'])

    # for stock in stocks:
    stock = 'aapl'
    for p in ps:
        for sharePer in sharePers:
            loss = 0.3  # Stop Loss
            model = 'Arima'
            startDate = '2017-09-05'
            startingCapital = 10000
            stop = '2018-02-05'

            result = trade(loss, model, p, sharePer, startDate, startingCapital, stop, stock)
            save_results(result, manager, stock)
