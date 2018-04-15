import pandas as pd
from mongoObjects import CollectionManager
from putAndGetData import avg_price_timeseries
import numpy as np
from utils import diff_multifeature


def create_sec_dic(stocks):
    sectorToStocks = {}
    for index,stock in stocks.iterrows():
        currentSector = stock['Sector']
        if currentSector not in sectorToStocks:
            sectorToStocks[currentSector] = [stock['Symbol']]
        else:
            sectorToStocks[currentSector] = sectorToStocks[currentSector] + [stock['Symbol']]
    return sectorToStocks

def get_data_by_sector(stocks,sector):
    dates = manager.dates()
    data = pd.DataFrame()
    for stock in stocks:
        stocksData = avg_price_timeseries(manager,stock.lower(),dates)
        if len(stocksData) != 1259:
            continue
        data[stock] = stocksData
    data.to_csv('sectorAnalysis/{0}.csv'.format(sector))

def get_data_to_files():

    sectorToStocks = create_sec_dic(stocks)

    for sector in sectors:
        get_data_by_sector(sectorToStocks[sector], sector)

def avg_data_files():
    avg = pd.DataFrame()
    for sector in sectors:
        data = pd.read_csv('sectorAnalysis/{0}.csv'.format(sector))
        avgs = []
        for i,day in data.iterrows():
            avgs.append(np.mean(day))
        avg[sector] = avgs
    avg.to_csv('sectorAnalysis/AveragePerformance.csv')

def differenced():
    data = pd.read_csv('sectorAnalysis/AveragePerformance.csv').iloc[:,1:]
    differenced = diff_multifeature(data)
    differenced.to_csv('sectorAnalysis/DifferencedAvg.csv')

def supervised_file():
    data = pd.read_csv('sectorAnalysis/DifferencedAvg.csv').iloc[:,1:]
    labels = pd.DataFrame()
    ys = []
    for i in range(0,len(data)-10):
        chunk = data.iloc[i:i+10,:].sum(axis=0)
        best = chunk.idxmax()
        ys.append(best)
    labels['day'] = range(len(ys))
    labels['y'] = ys
    labels.to_csv('sectorAnalysis/ys.csv')


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    stocks = pd.read_csv('stocks.csv')
    sectors = stocks['Sector'].unique()
    get_data_to_files()
    avg_data_files()
    differenced()
    supervised_file()









