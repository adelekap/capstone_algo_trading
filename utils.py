import matplotlib.pyplot as plt
import datetime as dt
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_3D(x,y,z):
    # xx, yy = np.meshgrid(x_range, x_range)
    # z = formula(x_range,intercept,coefficients)
    #
    # fig = plt.figure(figsize=(14,10))
    # ax = fig.gca(projection='3d')
    # ax.scatter(xs=y_train, zs=x_train['Water Maze CIPL'], ys=x_train['Working Memory CIPL'])
    # ax.plot_surface(X=z, Y=yy, Z=xx, color='r',alpha=0.5)
    # ax.set_ylabel('Working Memory CIPL')
    # ax.set_zlabel('Spatial Memory CIPL')
    # ax.set_xlabel('Age (months)')
    # props = dict(boxstyle='round', facecolor='g', alpha=0.5)
    # ax.text(0.05,0.95,1.0,'age = {0} + {2}(Working) + {1}(Spatial)\nCross Validation:{3}'.format(str(intercept.round(2)),
    #                                                                                              str(coefficients[0].round(2)),
    #                                                                                              str(coefficients[1].round(2)),
    #                                                                                              str(cv)),
    #         transform=ax.transAxes,fontsize=18,verticalalignment='top',bbox=props,horizontalalignment='left')
    # plt.savefig(dir+title+'.pdf')
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='b')
    plt.show()

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


