import datetime as dt
import sys
import math
import matplotlib.pyplot as plt
from DataHandler.mongoObjects import MongoDocument
import pandas as pd


def laterDate(date, j):
    """
    Gets the later date j days after
    :param date: date string YYYY-MM-DD
    :param j: number of days
    :return: later date
    """
    ds = [int(d) for d in date.split('-')]
    date = dt.datetime(ds[0], ds[1], ds[2])
    return str(date + dt.timedelta(days=j))[:10]


def diff_multifeature(dataset: pd.DataFrame, interval=1):
    """
    Differences a dataframe
    :param dataset: data
    :param interval: order of differencing
    :return: Differenced dataframe
    """
    differenced = pd.DataFrame()
    features = list(dataset.columns.values)
    for feature in features:
        series = dataset[feature]
        diff = list()
        for i in range(interval, len(series)):
            value = series[i] - series[i - interval]
            diff.append(value)
        differenced[feature] = diff
    return differenced


def split(timeseries: list, percent: float):
    """
    Splits the data into train and test sets
    :param timeseries: time series
    :param percent: percent of the length to put the split at
    :return: train set, test set
    """
    l = len(timeseries)
    index = round(l * percent)
    train = timeseries[:index]
    test = timeseries[index:]
    return train, test


def plot_capital(capital: list, time: list, stock: str, actual: list, percentGain='', drawdown='', possible='',
                 title='capital', model=''):
    """
    Plots the capital of a portfolio
    :param capital: the capital over time
    :param time: the dates
    :param stock: ticker
    :param actual: stock's price
    :param percentGain: percent change in trading period
    :param drawdown: MDD
    :param possible: possible return for 1 long position
    :param title: plot title
    :param model: model name
    :return: None
    """
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in time]
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(dates, capital, color='blue', label='Investor')
    axarr[0].set_title('Investments in {0}: {1}'.format(stock, percentGain, drawdown))
    axarr[1].plot(dates, actual, color='grey', label=stock + ' Price')
    axarr[1].set_title('Price of {0}: {1}%'.format(stock, possible))
    plt.xticks(fontsize=9, rotation=45)
    plt.tight_layout()
    plt.savefig('plots/{0}/{1}_{2}.png'.format(model, title, stock))
    print()
    print('COMPLETE')
    plt.show()


def MDD(series):
    """Maximum Drawdown (MDD) is an indicator of downside
    risk over a specified time period.
    MDD = (TroughValue – PeakValue) ÷ PeakValue
    """
    trough = min(series)
    peak = max(series)
    mdd = (peak - trough) / peak
    return round(mdd, 3)


def save_results(dict, manager, ticker):
    """
    Saves the results of trading in the database
    :param dict: results
    :param manager: collection manager
    :param ticker: stock ticker
    :return:
    """
    newDoc = MongoDocument(dict, ticker, [])
    manager.insert(newDoc)


class ProgressBar(object):
    def __init__(self, totalDays):
        self.totalDays = totalDays
        self.d = 1.0
        self.threshold = 0.05

    def initialize(self):
        print('||||||||||||||||||||||||||||||')
        print('...........PROGRESS...........')

    def progress(self):
        if self.d == self.totalDays:
            sys.stdout.flush()
        percent = self.d / self.totalDays
        if percent > self.threshold:
            sys.stdout.write('|||')
            self.threshold = math.ceil(percent * 10.0) / 10.0
        self.d += 1.0
        sys.stdout.flush()
