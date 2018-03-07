import matplotlib.pyplot as plt
import datetime as dt
import sys
import math


def laterDate(date, j):
    ds = [int(d) for d in date.split('-')]
    date = dt.datetime(ds[0], ds[1], ds[2])
    return date + dt.timedelta(days=j)


def split(timeseries: list, percent: float):
    l = len(timeseries)
    index = round(l * percent)
    train = timeseries[:index]
    test = timeseries[index:]
    return train, test


def plot_capital(capital: list, time: list, stock: str, actual: list, percentGain='', drawdown='', possible='',
                 title='capital'):
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in time]
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(dates, capital, color='blue', label='Investor')
    axarr[0].set_title('Investments in {0}: {1}  MDD={2}'.format(stock, percentGain, drawdown))
    axarr[1].plot(dates, actual, color='grey', label=stock + ' Price')
    axarr[1].set_title('Price of {0}: {1}%'.format(stock,possible))
    plt.xticks(fontsize=9, rotation=45)
    plt.tight_layout()
    plt.savefig('{0}.png'.format(title))
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


class ProgressBar(object):
    def __init__(self,totalDays):
        self.totalDays = totalDays
        self.d = 1.0
        self.threshold = 0.1

    def initialize(self):
        print('||||||||||||||||||||')
        print('......PROGRESS......')

    def progress(self):
        if self.d == self.totalDays:
            sys.stdout.write('||')
            sys.stdout.flush()
        percent = self.d/self.totalDays
        if percent > self.threshold:
            sys.stdout.write('||')
            self.threshold = math.ceil(percent * 10) / 10
        self.d += 1.0
        sys.stdout.flush()


