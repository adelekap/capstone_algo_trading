from mongoObjects import CollectionManager, FiveYearDocument
import apiCall as api
from pymongo import MongoClient
import pandas as pd

def add_5y_stock_data(ticker):
    """
    Adds 5-year technicals to the 'AlgoTradingDB' database
    for the specified stock.
    :param ticker: stock ticker
    :return: None
    """
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    stockData = api.iex_5y(ticker)
    for datapoint in stockData:
        newDoc = FiveYearDocument(datapoint, ticker)
        manager.insert(newDoc)


def add_data(function):
    """
    Adds data to the database for all of the S&P 500 stocks.
    :param function: the function that will add the data to the database
    :return: None
    """
    stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]
    for stock in stocks:
        try:
            function(stock)
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            print('Decoding JSON has failed: adding stock data for ' + stock + ' was unsuccessful!')



if __name__ == '__main__':
    add_data(add_5y_stock_data)