import matplotlib.pyplot as plt
import datetime as dt


def laterDate(date, j):
    ds = [int(d) for d in date.split('-')]
    date = dt.datetime(ds[0], ds[1], ds[2])
    return date + dt.timedelta(days=j)

def split(timeseries:list,percent:float):
    l = len(timeseries)
    index = round(l*percent)
    train = timeseries[:index]
    test = timeseries[index:]
    return train,test

def plot_capital(capital:list,time:list,stock:str,actual:list,percentGain='',percentPossible=''):
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in time]
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(dates,capital,color='blue',label='Investor')
    axarr[0].set_title('Investments in '+stock+' : '+str(percentGain))
    axarr[1].plot(dates,actual,color='grey',label=stock+' Price')
    axarr[1].set_title(stock+' Price : '+str(percentPossible)+'%')
    plt.xticks(fontsize=9, rotation=45)
    plt.savefig('capital.png')
    plt.show()

def MDD(series):
    """Maximum Drawdown (MDD) is an indicator of downside
    risk over a specified time period.
    MDD = (TroughValue – PeakValue) ÷ PeakValue
    """
    trough = min(series)
    peak = max(series)
    mdd = (trough-peak)/peak
    return mdd

def sharpe_ratio(series):
    """S(x) = (rx - Rf) / StdDev(x)"""
    pass