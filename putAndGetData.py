from mongoObjects import CollectionManager, MongoDocument
import apiCall as api
from pymongo import MongoClient
import pandas as pd
import datetime as dt

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


def create_timeseries(manager, ticker):
    """
    Creates a timeseries of a chain of opening
    and closing prices for specified stock.
    :param manager: collection manager
    :param ticker: string of ticker
    :return: timeseries
    """
    mmData = manager.find({'ticker': ticker})
    series = []
    dates = []
    for index, row in mmData.iterrows():
        series.append(row['open'])
        series.append(row['close'])
        dates.append(dt.datetime.strptime(row['date']+'-09',"%Y-%m-%d-%H"))
        dates.append(dt.datetime.strptime(row['date']+'-04',"%Y-%m-%d-%H"))
    return series,dates


def get_day_stats(manager,ticker,date):
    """
    Gets the close, high and low for a specified date.
    :param manager: collection manager
    :param ticker: string of ticker
    :param date: string of date 'YYYY-MM-DD'
    :return: (close, high, low)
    """
    mmData = manager.find({'ticker': ticker,'date':date})
    return mmData['close'][0],mmData['high'][0],mmData['low'][0]

def get_closes_highs_lows(manager,ticker):
    """
    Gets all of the closes, highs, and lows for specified stock.
    :param manager: collection manager
    :param ticker: string of ticker
    :return: (closes, highs, lows, dates)
    """
    mmData = manager.find({'ticker': ticker})
    closes = []
    highs = []
    lows = []
    dates = []
    for index, row in mmData.iterrows():
        closes.append(row['close'])
        highs.append(row['high'])
        lows.append(row['low'])
        dates.append(dt.datetime.strptime(row['date'], "%Y-%m-%d"))
    return closes,highs,lows,dates


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    add_data(manager,api.iex_5y,unwantedFields=['unadjustedVolume', 'change',
                                               'changePercent', 'label', 'changeOverTime'])

