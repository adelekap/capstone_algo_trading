from mongoObjects import CollectionManager, MongoDocument
import apiCall as api
from pymongo import MongoClient
import pandas as pd

def get_stock_data(manager, ticker, unwantedFields,function=api.iex_5y):
    """
    Adds 5-year technicals to the 'AlgoTradingDB' database
    for the specified stock.
    :param ticker: stock ticker
    :return: None
    """
    stockData = function(ticker)
    for datapoint in stockData:
        newDoc = MongoDocument(datapoint, ticker,unwantedFields)
        manager.insert(newDoc)


def add_data(manager,function,unwantedFields):
    """
    Adds data to the database for all of the S&P 500 stocks.
    :param function: the function that will add the data to the database
    :return: None
    """
    stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]
    for stock in stocks:
        try:
            get_stock_data(manager,stock,unwantedFields,function)
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            print('Decoding JSON has failed: adding stock data for ' + stock + ' was unsuccessful!')




if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    add_data(manager,api.iex_5y,unwantedFields=['unadjustedVolume', 'change',
                                               'changePercent', 'label', 'changeOverTime'])

