from DataHandler.mongoObjects import MongoDocument
import DataHandler.apiCall as api
import pandas as pd
import datetime as dt
import numpy as np


def get_stock_data(manager, ticker, unwantedFields, function=api.iex_5y):
    """
    Adds 5-year technicals to the 'AlgoTradingDB' database
    for the specified stock.
    :param ticker: stock ticker
    :return: None
    """
    stockData = function(ticker)
    for datapoint in stockData:
        newDoc = MongoDocument(datapoint, ticker, unwantedFields)
        manager.insert(newDoc)


def add_data(manager, function, unwantedFields):
    """
    Adds data to the database for all of the S&P 500 stocks.
    :param function: the function that will add the data to the database
    :return: None
    """
    stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]
    for stock in stocks:
        try:
            get_stock_data(manager, stock, unwantedFields, function)
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            print('Decoding JSON has failed: adding stock data for ' + stock + ' was unsuccessful!')


def create_timeseries(manager, ticker:str):
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
        dates.append(dt.datetime.strptime(row['date'] + '-09', "%Y-%m-%d-%H"))
        dates.append(dt.datetime.strptime(row['date'] + '-04', "%Y-%m-%d-%H"))
    return series, dates


def get_multifeature_data(manager, ticker:str):
    """
    Gets data for multivariate regression
    :param manager: collection manager for the 5 year technicals
    :param ticker: stock ticker
    :return: data for the close, high, low, open, vwap, and the volume
    """
    data = manager.find({'ticker': ticker})[['close', 'high', 'low', 'open', 'vwap', 'volume']]
    return data


def get_day_stats(manager, ticker:str, date:str):
    """
    Gets the close, high and low for a specified date.
    :param manager: collection manager
    :param ticker: string of ticker
    :param date: string of date 'YYYY-MM-DD'
    :return: (close, high, low)
    """
    mmData = manager.find({'ticker': ticker, 'date': date})
    return mmData['close'][0], mmData['high'][0], mmData['low'][0]


def get_closes_highs_lows(manager, ticker:str):
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
    return closes, highs, lows, dates


def avg_price_timeseries(manager, ticker, dates):
    """
    Calculates avg of the data
    :param manager: collection manager for the 5 year technicals
    :param ticker: stock ticker
    :param dates: dates of trading
    :return: averaged time series
    """
    series = []
    for date in dates:
        data = manager.find({'ticker': ticker, 'date': date})
        c = data['close'][0]
        h = data['high'][0]
        l = data['low'][0]
        avg = (c + h + l) / 3.0
        series.append(avg)
    return series


def rel_volume(manager, ticker, date):
    """
    Returns the relative volume of the day for the given stock
    :param manager: collection manager for 5 year technicals
    :param ticker: stock ticker
    :param date: date of interest
    :return: relative volume
    """
    allVol = np.mean(manager.find_distinct({'ticker': ticker}, 'volume'))
    vol = manager.find({'ticker': ticker, 'date': date})['volume'][0]
    return vol / allVol
