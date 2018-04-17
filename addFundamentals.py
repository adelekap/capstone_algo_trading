import json
import pandas as pd
from mongoObjects import CollectionManager
from datetime import date
import datetime
import datedelta
import numpy as np
import calendar
from pandas.tseries.offsets import BDay

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


def next_quarter(current: str):
    year = int(current.split(' ')[1])
    q = int(current[1])
    if q < 4:
        q += 1
    else:
        year += 1
        q = 1
    return f'Q{q} {year}'


def get_next_business_day(current_date: date):
    day_of_week = current_date.weekday()
    next_business_day = current_date

    if day_of_week > 4:
        next_business_day = current_date + (datedelta.DAY * (7 - day_of_week))

    return next_business_day

def get_next_day_for_sure(price_getter, day: date):
    business_day = get_next_business_day(day)
    day_as_string = str(business_day)
    result = price_getter(day_as_string)

    while result.shape[0] == 0:
        business_day += datedelta.DAY
        day_as_string = str(business_day)
        result = price_getter(day_as_string)

    return result

def calculate_performance(ticker, date1: date, date2: date):
    ticker = ticker.lower()
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')

    date1 = get_next_business_day(date1)
    date2 = get_next_business_day(date2)

    price_1_getter = lambda d : manager.find({'ticker': ticker, 'date': d})

    # price1 = manager.find({'ticker': ticker, 'date': str(get_next_business_day(date1))})
    price1 = get_next_day_for_sure(price_getter=price_1_getter, day=date1)['vwap'][0]
    # if price1.shape[0] == 0:
    #     price1 = manager.find({'ticker':ticker,'date':str(get_next_business_day(date1) + datetime.timedelta(days=1))})
    # price1 = price1['vwap'][0]
    price2 = get_next_day_for_sure(price_getter=price_1_getter, day=date2)['vwap'][0]
    # if price2.shape[0] == 0:
    #     price2 = manager.find({'ticker':ticker,'date':str(get_next_business_day(date2)+ datetime.timedelta(days=1))})
    # price2 = price2['vwap'][0]
    manager.close()
    return (price2 - price1) / price2


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


def get_all_future_quarters(today_date: date):  # feb 5th 2018 - today's date
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


def get_all_fundamentals(tickers: list, quarters: list,final=False):
    manager = CollectionManager('10y_Fundamentals', 'AlgoTradingDB')
    allFundamentals = []
    performances = []
    for quarter in quarters:
        quarterly = []
        for ticker in tickers:
            data = manager.find({'ticker': ticker.upper(), 'quarter': quarter}).iloc[:, :-2]
            if len(data) == 0:
                continue
            announcementDate = data.iloc[:, -1][0].date()
            data = data.iloc[:, :-2]
            if final:
                nextAnnouncementDate = '2018-02-05'
            else:
                nextAnnouncementDate = manager.find({'ticker': ticker.upper(),
                                                 'quarter': next_quarter(quarter)})['date'][0].date()
            performance = calculate_performance(ticker, announcementDate, nextAnnouncementDate)
            performances.append(performance)
            quarterly.append(data.values[0])
        allFundamentals.append(quarterly)
    return np.array(allFundamentals), performances


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
    # get_fundamental_data(['aapl', 'bac'], '2018-01-01')
    print(get_next_business_day(date(2013, 3, 31)))