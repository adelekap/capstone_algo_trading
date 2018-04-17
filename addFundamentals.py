import json
import pandas as pd
from mongoObjects import CollectionManager
from datetime import date
import datetime
import datedelta
import numpy as np

quarters = {1: "Q4", 2: "Q4", 3: "Q4",
            4: "Q1", 5: "Q1", 6: "Q1",
            7: "Q2", 8: "Q2", 9: "Q2",
            10: "Q3", 11: "Q3", 12: "Q3"}


def get_quarter(dateObject: date) -> str:
    month = dateObject.month
    current = quarters[month] + " " + str(dateObject.year if quarters[month] != "Q4" else dateObject.year - 1)
    previous = quarters[month - 3 if month >= 4 else (month - 3) + 12] + " " + \
               str(dateObject.year if quarters[month] != "Q4" else dateObject.year - 1)
    return current, previous


def months_apart(earlier_date: date, later_date: date) -> int:
    return (later_date.year - earlier_date.year) * 12 + later_date.month - earlier_date.month


def get_all_past_quarters(today_date: date):  # feb 5th 2013 - today's date
    historical_date = date(2013, 1, 1)

    if today_date < historical_date or today_date > date(2018, 2, 5):
        return []

    quarters_apart = months_apart(historical_date, today_date) // 3
    available_quarters = [get_quarter(historical_date)[0]]

    for i in range(quarters_apart):
        historical_date += (datedelta.MONTH * 3)
        available_quarters.append(get_quarter(historical_date)[0])

    return available_quarters


def get_all_future_quarters(today_date: date): # feb 5th 2018 - today's date
    future_date = date(2018, 2, 5)

    if today_date > future_date:
        return []

    quarters_apart = months_apart(future_date, today_date) // 3
    available_quarters = [get_quarter(future_date)[0]]

    for i in range(quarters_apart):
        future_date -= (datedelta.MONTH * 3)
        available_quarters.append(get_quarter(future_date)[0])

    return available_quarters

def add_fundamentals_to_db():
    """
    Adds the fundamental data to the database
    :return:
    """
    fundFile = 'sectorAnalysis/fundamentals/combinedFundamentals.json'
    funds = pd.read_json(fundFile)

    manager = CollectionManager('10y_Fundamentals', 'AlgoTradingDB')

    for index, row in funds.iterrows():
        document = row.to_dict()
        manager.insert(document, is_dictionary=True)


def get_all_fundamentals(tickers: list, quarters: list):
    manager = CollectionManager('10y_Fundamentals', 'AlgoTradingDB')
    allFundamentals = []
    performances = []
    for quarter in quarters:
        quarterly = []
        for ticker in tickers:
            data = manager.find({'ticker': ticker.upper(), 'quarter': quarter}).iloc[:, :-4]
            if len(data) == 0:
                continue
            
            quarterly.append(data.values[0])
        allFundamentals.append(quarterly)
    return np.array(allFundamentals)


def get_fundamental_data(tickers: list, dateString):
    """
    Gets fundamental data from the database for the
    stocks in the list of tickers.
    """
    manager = CollectionManager('10y_Fundamentals', 'AlgoTradingDB')
    tradeDay = datetime.datetime.strptime(dateString, '%Y-%m-%d').date()
    current, previous = get_quarter(tradeDay)
    quarterlyFundamentals = pd.DataFrame()
    for ticker in tickers:
        data = manager.find({'ticker': ticker.upper(), 'quarter': current}).iloc[:, :-4]
        if len(data) == 0:
            data = manager.find({'ticker': ticker.upper(), 'quarter': previous}).iloc[:, :-4]
            if len(data) == 0:
                continue
        data['ticker'] = tickers.index(ticker)
        if len(quarterlyFundamentals) == 0:
            quarterlyFundamentals = data
        else:
            quarterlyFundamentals = pd.concat([quarterlyFundamentals, data])

    return quarterlyFundamentals


if __name__ == '__main__':
    get_fundamental_data(['aapl', 'bac'], '2018-01-01')
