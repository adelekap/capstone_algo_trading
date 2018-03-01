import datetime


def laterDate(date, j):
    ds = [int(d) for d in date.split('-')]
    date = datetime.datetime(ds[0], ds[1], ds[2])
    return date + datetime.timedelta(days=j)

def split(timeseries:list,percent:float):
    l = len(timeseries)
    index = round(l*percent)
    train = timeseries[:index]
    test = timeseries[index:]
    return train,test