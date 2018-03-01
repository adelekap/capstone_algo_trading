

def split(timeseries:list,percent:float):
    l = len(timeseries)
    index = round(l*percent)
    train = timeseries[:index]
    test = timeseries[index:]
    return train,test