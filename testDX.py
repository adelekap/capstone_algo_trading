import requests
from json import dumps


def iex_call(url,parameters=None):
    return requests.get('https://api.iextrading.com/1.0'+url).json()


if __name__ == '__main__':
    aaplStats = iex_call('/stock/aapl/stats')
    # print(dumps(aaplStats,indent=2))
    aaplOneDayChart = iex_call('/stock/aapl/financials')
    print(dumps(aaplOneDayChart,indent=2))

    # DOWNLOADED SIMFIN DATASET
    #data = pd.read_csv('simfin-data.csv',skiprows=1,nrows=59)
