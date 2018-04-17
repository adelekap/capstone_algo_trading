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

features = ['Asset Growth','Book Value per Share Growth','Debt Growth','Dividends per Basic Common Share Growth',
            'EBIT Growth','EPS Diluted Growth','EPS Growth','Gross Profit Growth','Inventory Growth','Net Income Growth',
            'Operating Cash Flow Growth','Trade and Non-Trade Receivables Growth','Weighted Average Shares Diluted Growth',
            'quarter']


def add_fundamentals_to_db():
    """
    Adds the fundamental data to the database from a json file
    :return:None
    """
    fundFile = 'sectorAnalysis/fundamentals/combinedFundamentals.json'
    funds = pd.read_json(fundFile)

    manager = CollectionManager('10y_Fundamentals', 'AlgoTradingDB')

    for index, row in funds.iterrows():
        document = row.to_dict()
        manager.insert(document, is_dictionary=True)
    manager.close()

def get_quarter(dateObject: date):
    """
    Gets the current quarter and the previous quarter at some date.
    :param dateObject: date
    :return: current quarter and next quarter (tuple)
    """
    month = dateObject.month
    current = quarters[month] + " " + str(dateObject.year if quarters[month] != "Q4" else dateObject.year - 1)
    previous = quarters[month - 3 if month >= 4 else (month - 3) + 12] + " " + \
               str(dateObject.year if quarters[month] != "Q4" else dateObject.year - 1)
    return current, previous


def next_quarter(current: str):
    """
    Gets the next quarter given some date
    :param current: current quarter
    :return: next quarter
    """
    year = int(current.split(' ')[1])
    q = int(current[1])
    if q < 4:
        q += 1
    else:
        year += 1
        q = 1
    return f'Q{q} {year}'


def get_next_business_day(current_date: date):
    """
    From some day, returns the next business day.
    :param current_date:
    :return: next business day date
    """
    day_of_week = current_date.weekday()
    next_business_day = current_date

    if day_of_week > 4:
        next_business_day = current_date + (datedelta.DAY * (7 - day_of_week))

    return next_business_day

def get_next_trading_day(dates, days: list):
    """
    Ensures that the day was a trading day and gets the data
    for that day.
    :param price_getter: lambda function
    :param day: date
    :return: price for that day of trading
    """
    newDates = []
    for day in days:
        day = day
        while day not in dates:
            day += datedelta.DAY
        newDates.append(day)
    return newDates


def calculate_performance(ticker, dates1: list, dates2: list):
    """
    Gets how much the stock has changed since the last
    quarter reportings.
    :param ticker: stock ticker
    :param date1: beginining of the quarter
    :param date2: end of the quarter
    :return: percent change in stock price
    """
    ticker = ticker.lower()
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')

    prices = manager.find({'ticker': ticker})
    dates = [datetime.datetime.strptime(priceDate,'%Y-%m-%d').date() for priceDate in prices['date']]

    pricesStart = [prices[prices['date']==str(d1)]['vwap'].values[0] for d1 in get_next_trading_day(dates,dates1)]
    pricesEnd = [prices[prices['date']==str(d2)]['vwap'].values[0] for d2 in get_next_trading_day(dates,dates2)]
    manager.close()

    performances = [((p[0]-p[1])/p[0]) for p in zip(pricesStart,pricesEnd)]
    return performances


def months_apart(earlier_date: date, later_date: date) -> int:
    return (later_date.year - earlier_date.year) * 12 + later_date.month - earlier_date.month


def get_all_past_quarters(today_date: date):  # feb 5th 2013 - today's date
    """
    Gets a list of all the past quarters
    :param today_date: date
    :return: list of quarters
    """
    historical_date = date(2013, 1, 1)

    if today_date < historical_date or today_date > date(2018, 2, 5):
        return []

    quarters_apart = months_apart(historical_date, today_date) // 3
    available_quarters = [get_quarter(historical_date)[0]]

    for i in range(quarters_apart):
        historical_date += (datedelta.MONTH * 3)
        available_quarters.append(get_quarter(historical_date)[0])
    return available_quarters

def find_best_stock(performances:list):
    best = []
    for quarter in range(len(performances[0])):
        q = [abs(x[quarter]) for x in performances]
        best.append(np.argmax(q))
    return best

def get_all_future_quarters(today_date: date):
    """
    Gets all of the potential future quarters until 02-06-2018
    :param today_date: date
    :return: list of quarters
    """
    future_date = date(2018, 2, 5)

    if today_date > future_date:
        return []

    quarters_apart = months_apart(future_date, today_date) // 3
    available_quarters = [get_quarter(future_date)[0]]

    for i in range(quarters_apart):
        future_date -= (datedelta.MONTH * 3)
        available_quarters.append(get_quarter(future_date)[0])
    return available_quarters


# def get_all_fundamentals(tickers: list, quarters: list,final=False):
#     """
#     Gets all of the fundamentals for a list of tickers and list of quarters
#     :param tickers: stocks
#     :param quarters: list of quarters
#     :param final: whether this is a final prediction
#     :return: Xs and ys
#     """
#     manager = CollectionManager('10y_Fundamentals', 'AlgoTradingDB')
#     allFundamentals = []
#     performances = []
#     for quarter in quarters:
#         quarterly = []
#         for ticker in tickers:
#             data = manager.find({'ticker': ticker.upper(), 'quarter': quarter}).iloc[:, :-2]
#             if len(data) == 0:
#                 continue
#             announcementDate = data.iloc[:, -1][0].date()
#             data = data.iloc[:, :-2]
#             if final:
#                 nextAnnouncementDate = '2018-02-05'
#             else:
#                 nextAnnouncementDate = manager.find({'ticker': ticker.upper(),
#                                                  'quarter': next_quarter(quarter)})['date'][0].date()
#             performance = calculate_performance(ticker, announcementDate, nextAnnouncementDate)
#             performances.append(performance)
#             quarterly.append(data.values[0])
#         allFundamentals.append(quarterly)
#     manager.close()
#     return np.array(allFundamentals), performances

def sort_quarters(dataframe,quarters):
    if len(dataframe) == 0:
        return dataframe
    dataframe['quarter'] = [quarters.index(v) for v in list(dataframe['quarter'])]
    dataframe = dataframe.sort_values('quarter')
    return dataframe

def get_all_fundamentals(tickers: list, quarters: list,final=False):
    """
    Gets all of the fundamentals for a list of tickers and list of quarters
    :param tickers: stocks
    :param quarters: list of quarters
    :param final: whether this is a final prediction
    :return: Xs and ys
    """
    manager = CollectionManager('10y_Fundamentals', 'AlgoTradingDB')
    allFundamentals = pd.DataFrame()
    performances = []
    for ticker in tickers:
        data = sort_quarters(manager.find({'ticker': ticker.upper(), 'quarter': {"$in":quarters}}),quarters)
        if len(data) == 0:
            continue
        announcementDates = [d.date() for d in data.iloc[:, -3]]
        nextAnnouncementDates = announcementDates[1:] + [datetime.datetime.strptime('2018-02-05','%Y-%m-%d').date()]
        data = data[features]

        performance = calculate_performance(ticker, announcementDates, nextAnnouncementDates)
        performances.append(performance)
        for index,funds in data.iterrows():
            tempDF = pd.DataFrame()
            tempDF['fundamentals'] = list(funds)[:-1]
            tempDF['ticker'] = [ticker for i in range(len(funds)-1)]
            tempDF['quarter'] = [index for i in range(len(funds)-1)]
            allFundamentals = pd.concat([allFundamentals,tempDF])
    manager.close()

    trainingData = []
    for quarter in range(len(quarters)):
        q = []
        for ticker in tickers:
            tickerdata = allFundamentals[allFundamentals['ticker'] == ticker]
            quarterdata = list(tickerdata[tickerdata['quarter'] == quarter]['fundamentals'])
            q.append(np.array(quarterdata))
        trainingData.append(np.array(q))
    trainingDataX = np.array(trainingData)
    trainingDataY = np.array(find_best_stock(performances))
    return trainingDataX, trainingDataY.reshape(len(quarters),1)