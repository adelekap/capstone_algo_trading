import requests
import json


def iex_call(stock, dataType):
    """
    Generic executution of an API call to IEX Trading for the specified stock.
    :param stock: string of stock ticker -- all lowercase
    :param dataType: string to data type source (e.g. 'chart/5y')
    :return: JSON object
    """
    baseUrl = 'https://api.iextrading.com/1.0/stock/'
    return requests.get('{0}{1}/{2}'.format(baseUrl, stock, dataType)).text


def iex_5y(stock: str):
    """
    Gets 5 year historical data for specified stock.
    :param stock: string of stock ticker
    :return: List of dictionaries with the data for each day.
    """
    result = json.loads(iex_call(stock.lower(), 'chart/5y'))
    return result


def iex_1d(stock:str):
    """
    Gets 1 day stock data for specified stock.
    :param stock: string of stock ticher
    :return: List of dictionaries with the data for each minute.
    """
    result = json.loads(iex_call(stock.lower(), 'chart/1d'))
    return result
