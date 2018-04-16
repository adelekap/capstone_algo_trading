import matplotlib.pyplot as plt
import datetime as dt
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import seaborn as sns
from mongoObjects import MongoDocument
import pandas as pd

def laterDate(date, j):
    ds = [int(d) for d in date.split('-')]
    date = dt.datetime(ds[0], ds[1], ds[2])
    return str(date + dt.timedelta(days=j))[:10]

def diff_multifeature(dataset,interval=1):
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
    l = len(timeseries)
    index = round(l * percent)
    train = timeseries[:index]
    test = timeseries[index:]
    return train, test


def plot_capital(capital: list, time: list, stock: str, actual: list, percentGain='', drawdown='', possible='',
                 title='capital',model=''):
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in time]
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(dates, capital, color='blue', label='Investor')
    axarr[0].set_title('Investments in {0}: {1}'.format(stock, percentGain, drawdown))
    axarr[1].plot(dates, actual, color='grey', label=stock + ' Price')
    axarr[1].set_title('Price of {0}: {1}%'.format(stock,possible))
    plt.xticks(fontsize=9, rotation=45)
    plt.tight_layout()
    plt.savefig('plots/{0}/{1}_{2}.png'.format(model,title,stock))
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
    return round(mdd,3)


def sharpe_ratio(series):
    """S(x) = (rx - Rf) / StdDev(x)"""
    pass

def plot_3D(x,y,z,title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('p')
    plt.ylabel('sharePer')
    plt.title(title)
    ax.scatter3D(x, y, z, color='b')
    plt.show()

def save_results(dict, manager, ticker):
    newDoc = MongoDocument(dict, ticker, [])
    manager.insert(newDoc)

class ProgressBar(object):
    def __init__(self,totalDays):
        self.totalDays = totalDays
        self.d = 1.0
        self.threshold = 0.05

    def initialize(self):
        print('||||||||||||||||||||||||||||||')
        print('...........PROGRESS...........')

    def progress(self):
        if self.d == self.totalDays:
            sys.stdout.flush()
        percent = self.d/self.totalDays
        if percent > self.threshold:
            sys.stdout.write('|||')
            self.threshold = math.ceil(percent * 10.0) / 10.0
        self.d += 1.0
        sys.stdout.flush()
