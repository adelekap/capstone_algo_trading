import pandas as pd
from mongoObjects import CollectionManager
from datetime import date
from datetime import datetime as dt
import datedelta
import numpy as np

features = ['Asset Growth', 'Book Value per Share Growth', 'Debt Growth', 'Dividends per Basic Common Share Growth',
            'EBIT Growth', 'EPS Diluted Growth', 'EPS Growth', 'Gross Profit Growth', 'Inventory Growth',
            'Net Income Growth',
            'Operating Cash Flow Growth', 'Trade and Non-Trade Receivables Growth']


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


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, dt):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))


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
    dates = [dt.strptime(priceDate, '%Y-%m-%d').date() for priceDate in prices['date']]

    pricesStart = [prices[prices['date'] == str(d1)]['vwap'].values[0] for d1 in get_next_trading_day(dates, dates1)]
    pricesEnd = [prices[prices['date'] == str(d2)]['vwap'].values[0] for d2 in get_next_trading_day(dates, dates2)]
    manager.close()

    performances = [((p[0] - p[1]) / p[0]) for p in zip(pricesStart, pricesEnd)]
    return performances


def get_historical_fundamentals(ticker: str, d:date, manager:CollectionManager):
    current_day = dt(d.year,d.month,d.day)
    allTickersFundamentals = manager.find({'ticker':ticker,'date':{'$lte':current_day}}).sort_values('date')
    test = manager.find({'ticker':ticker,'date':{'$gte':current_day}}).sort_values('date')

    return allTickersFundamentals[features], [announce.date() for announce in allTickersFundamentals['date'].tolist()],test


def find_best_stock(performances: pd.DataFrame):
    best = []
    stocks = performances.columns.values
    for index, values in performances.iterrows():
        maximum = np.argmax([abs(v) for v in values])
        stock = stocks[maximum]
        best.append(stock)
    return best


def get_all_fundamentals(stocks: list, tradeDate:date):
    """
    Gets all of the fundamentals for a list of tickers and list of quarters
    :param tickers: stocks
    :param quarters: list of quarters
    :param final: whether this is a final prediction
    :return: Xs and ys
    """
    manager = CollectionManager('10y_Fundamentals', 'AlgoTradingDB')
    tickers_set = set(stocks)
    all_fundamental_tickers = set(manager.find({})["ticker"])
    tickers = list(tickers_set.intersection(all_fundamental_tickers))

    allFundamentals = pd.DataFrame()
    performances = pd.DataFrame()
    quarters = 17
    allTest = pd.DataFrame()
    testDates = []
    for ticker in tickers:
        data, announcementDates,test = get_historical_fundamentals(ticker, tradeDate, manager)
        nextAnnouncementDates = announcementDates[1:] + [dt.strptime('2018-02-05', '%Y-%m-%d').date()]

        performance = calculate_performance(ticker, announcementDates, nextAnnouncementDates)
        if len(testDates) == 0:
            testDates = test['date'].tolist()

        if len(performance) != 17:
            performance = performance[len(performance)-17:]
            performances[ticker] = performance
        else:
            performances[ticker] = performance

        for index, funds in data.iterrows():
            tempDF = pd.DataFrame()
            tempDF['fundamentals'] = list(funds)[:-1]
            tempDF['ticker'] = [ticker for i in range(len(funds) - 1)]
            tempDF['quarter'] = [index for j in range(len(funds) - 1)]
            allFundamentals = pd.concat([allFundamentals, tempDF])

        for index, testFunds in test.iterrows():
            temp = pd.DataFrame()
            temp['fundamentals'] = list(testFunds)[:-1]
            temp['ticker'] = [ticker for k in range(len(testFunds) - 1)]
            temp['quarter'] = [index for l in range(len(testFunds) - 1)]
            allTest = pd.concat([allTest, temp])

    manager.close()

    trainingData = []
    for quarter in range(quarters):
        q = []
        for ticker in tickers:
            tickerdata = allFundamentals[allFundamentals['ticker'] == ticker]
            quarterdata = tickerdata[tickerdata['quarter'] == quarter]['fundamentals']
            q.append(quarterdata.tolist())
        trainingData.append(np.array(q))
    trainingDataX = np.array(trainingData)
    trainingDataY = find_best_stock(performances)

    allTestY = []
    quarterLen = len(allTest['quarter'].unique())
    if quarterLen == 1:
        fix = allTest.copy()
        fix['quarter'] = [2 for i in range(len(fix))]
        allTest = pd.concat([allTest,fix])
    for testQuarter in range(2):
        testQ = []
        for tick in tickers:
            tickData = allTest[allTest['ticker'] == tick]
            testQuarterData = tickData[tickData['quarter'] == testQuarter]['fundamentals']
            if testQuarterData.shape[0] != 15:
                print('ERROR ' + tick)
            testQ.append(testQuarterData.tolist()[:-4])
        allTestY.append(np.array(testQ))

    return trainingDataX, trainingDataY, np.array(allTestY), testDates, tickers

if __name__ == '__main__':
    print(get_all_fundamentals(['RHT'],date(2017,9,7)))