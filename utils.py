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

def plot_capital(capital:list,time:list,stock:str):
    dates = [dt.datetime.strptime(d, "%Y-%m-%d-%H") for d in time]
    plt.plot(dates,capital)
    plt.title('Capital -- investments in '+stock)
    plt.savefig('capital.png')
    plt.show()